"""Orderbook snapshot collector for real data backtesting.

Collects orderbook snapshots every N seconds for selected markets.
Stores data in SQLite for later analysis and backtesting.

Usage:
    python3 -m mantis.collector              # collect top 10 markets
    python3 -m mantis.collector --top 20     # collect top 20
    python3 -m mantis.collector --interval 15  # every 15 seconds
"""
from __future__ import annotations

import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .polymarket_client import PolymarketClient
from .scanner import compute_total_q_min

logger = logging.getLogger("mantis.collector")

DB_PATH = Path("/workspace/mantis/data/snapshots.db")


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT,
            token_id TEXT,
            daily_rate REAL,
            min_size INTEGER,
            max_spread REAL,
            first_seen TEXT,
            last_seen TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ts_unix REAL NOT NULL,
            condition_id TEXT NOT NULL,
            token_id TEXT NOT NULL,
            best_bid REAL,
            best_ask REAL,
            midpoint REAL,
            spread_cents REAL,
            bid_depth_5 REAL,
            ask_depth_5 REAL,
            total_q REAL,
            n_bid_levels INTEGER,
            n_ask_levels INTEGER,
            bids_json TEXT,
            asks_json TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(condition_id, ts_unix)
    """)
    conn.commit()
    return conn


def select_markets(client: PolymarketClient, config, top_n: int = 10):
    """Select top markets by reward pool size (lightweight, no full scan).

    Only fetches orderbooks for the top candidates, not all 1000+.
    """
    cfg = config.scanner
    all_markets = client.fetch_all_sampling_markets()

    # Filter by basic criteria
    candidates = []
    for m in all_markets:
        if not m.active or not m.yes_token:
            continue
        if m.rewards.daily_rate < cfg.min_reward_rate:
            continue
        if m.rewards.min_size > cfg.max_min_size:
            continue
        price = m.yes_token.price
        if not (cfg.min_price <= price <= cfg.max_price):
            continue
        candidates.append(m)

    # Sort by daily_rate descending, take top N*2 to have room for failures
    candidates.sort(key=lambda m: -m.rewards.daily_rate)
    candidates = candidates[:top_n * 3]

    logger.info(f"Light select: {len(candidates)} candidates, fetching orderbooks...")

    from .scanner import estimate_your_q

    selected = []
    for m in candidates:
        if len(selected) >= top_n:
            break
        try:
            token = m.yes_token
            ob = client.fetch_orderbook(token.token_id)
            if not ob.bids or not ob.asks:
                continue

            total_q = compute_total_q_min(ob, m.rewards.max_spread)
            if total_q < 1.0:
                total_q = 1.0

            half_sp = (m.rewards.max_spread * config.engine.reward_spread_pct) / 100
            half_sp = max(half_sp, 0.001)
            your_q = estimate_your_q(
                m.rewards.min_size * ob.midpoint,
                ob.midpoint, half_sp, m.rewards.max_spread,
            )
            cap_needed = m.rewards.min_size * ob.midpoint * 2
            your_share = your_q / (total_q + your_q)
            est_daily = m.rewards.daily_rate * your_share
            rpc = est_daily / cap_needed if cap_needed > 0 else 0

            selected.append({
                "condition_id": m.condition_id,
                "question": m.question,
                "token_id": token.token_id,
                "daily_rate": m.rewards.daily_rate,
                "min_size": m.rewards.min_size,
                "max_spread": m.rewards.max_spread,
                "est_daily": est_daily,
                "reward_per_capital": rpc,
            })
            time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Skip {m.question[:30]}: {e}")
            continue

    # Re-sort by reward_per_capital
    selected.sort(key=lambda x: -x["reward_per_capital"])
    return selected


def save_market(conn: sqlite3.Connection, m: dict):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO markets (condition_id, question, token_id, daily_rate, min_size, max_spread, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(condition_id) DO UPDATE SET last_seen=?, daily_rate=?
    """, (m["condition_id"], m["question"], m["token_id"],
          m["daily_rate"], m["min_size"], m["max_spread"], now, now, now, m["daily_rate"]))
    conn.commit()


def collect_snapshot(client: PolymarketClient, conn: sqlite3.Connection, m: dict) -> bool:
    """Collect one orderbook snapshot. Returns True if successful."""
    try:
        ob = client.fetch_orderbook(m["token_id"])
        if not ob.bids and not ob.asks:
            return False

        now = datetime.now(timezone.utc)
        total_q = compute_total_q_min(ob, m["max_spread"])

        bid_depth_5 = sum(l.size for l in ob.bids[:5])
        ask_depth_5 = sum(l.size for l in ob.asks[:5])

        bids_json = json.dumps([{"p": l.price, "s": l.size} for l in ob.bids[:10]])
        asks_json = json.dumps([{"p": l.price, "s": l.size} for l in ob.asks[:10]])

        conn.execute("""
            INSERT INTO snapshots
            (ts, ts_unix, condition_id, token_id, best_bid, best_ask, midpoint,
             spread_cents, bid_depth_5, ask_depth_5, total_q, n_bid_levels, n_ask_levels,
             bids_json, asks_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.isoformat(), now.timestamp(), m["condition_id"], m["token_id"],
            ob.best_bid, ob.best_ask, ob.midpoint, ob.spread_cents,
            bid_depth_5, ask_depth_5, total_q,
            len(ob.bids), len(ob.asks), bids_json, asks_json,
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.debug(f"Snapshot failed for {m['condition_id'][:8]}: {e}")
        return False


def run_collector(top_n: int = 10, interval: int = 30, config_path: str = "config.yaml"):
    """Main collector loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    config = load_config(config_path)
    client = PolymarketClient()
    conn = init_db()

    # Graceful shutdown
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
        logger.info("Shutdown signal received, finishing current cycle...")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Select markets
        logger.info(f"Selecting top {top_n} markets...")
        markets = select_markets(client, config, top_n)
        logger.info(f"Tracking {len(markets)} markets:")
        for i, m in enumerate(markets):
            save_market(conn, m)
            logger.info(f"  #{i+1}: ${m['daily_rate']}/d | rpc={m['reward_per_capital']:.2f} | {m['question'][:50]}")

        # Collection loop
        cycle = 0
        total_snaps = 0
        logger.info(f"Starting collection: {len(markets)} markets, every {interval}s")
        logger.info(f"Data saved to: {DB_PATH}")

        while running:
            cycle += 1
            t0 = time.time()
            ok = 0
            for m in markets:
                if not running:
                    break
                if collect_snapshot(client, conn, m):
                    ok += 1
                time.sleep(0.05)  # rate limit

            total_snaps += ok
            elapsed = time.time() - t0
            if cycle % 10 == 1:  # log every 10th cycle
                logger.info(
                    f"Cycle {cycle}: {ok}/{len(markets)} snapshots in {elapsed:.1f}s | "
                    f"total: {total_snaps} snapshots"
                )

            # Wait for next cycle
            wait = max(0, interval - elapsed)
            if wait > 0 and running:
                time.sleep(wait)

        # Re-scan every hour to pick up new markets
        # (not implemented yet - would need a timer check)

    finally:
        client.close()
        conn.close()
        logger.info(f"Collector stopped. Total snapshots: {total_snaps}")


def get_stats(db_path: Path = DB_PATH) -> str:
    """Get collection stats as text."""
    if not db_path.exists():
        return "No data collected yet."

    conn = sqlite3.connect(str(db_path))
    lines = []

    # Overall stats
    row = conn.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM snapshots").fetchone()
    total, first_ts, last_ts = row
    lines.append(f"Total snapshots: {total}")
    lines.append(f"First: {first_ts}")
    lines.append(f"Last:  {last_ts}")
    lines.append("")

    # Per market
    rows = conn.execute("""
        SELECT m.question, s.condition_id, COUNT(*) as cnt,
               MIN(s.midpoint) as min_mid, MAX(s.midpoint) as max_mid,
               AVG(s.spread_cents) as avg_spread, AVG(s.total_q) as avg_q
        FROM snapshots s
        JOIN markets m ON s.condition_id = m.condition_id
        GROUP BY s.condition_id
        ORDER BY cnt DESC
    """).fetchall()

    lines.append(f"Markets tracked: {len(rows)}")
    lines.append("")
    for q, cid, cnt, min_mid, max_mid, avg_spread, avg_q in rows:
        lines.append(f"  {q[:50]}")
        lines.append(f"    Snapshots: {cnt} | Mid: {min_mid:.3f}-{max_mid:.3f} | Spread: {avg_spread:.1f}c | Q: {avg_q:.0f}")

    conn.close()
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mantis Orderbook Collector")
    parser.add_argument("--top", type=int, default=10, help="Number of markets to track")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between snapshots")
    parser.add_argument("--stats", action="store_true", help="Show collection stats and exit")
    args = parser.parse_args()

    if args.stats:
        print(get_stats())
    else:
        import os
        os.chdir("/workspace/mantis")
        run_collector(top_n=args.top, interval=args.interval)
