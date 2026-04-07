"""Enhanced orderbook and trade collector for realistic backtesting.

Collects:
- Orderbook snapshots every N seconds
- Trade flow from public API (for VPIN calculation)
- Orderbook change events (for queue dynamics)
- Q competition tracking over time

Usage:
    python3 -m mantis.collector              # collect top 10 markets
    python3 -m mantis.collector --top 20     # collect top 20
    python3 -m mantis.collector --interval 15  # every 15 seconds
    python3 -m mantis.collector --trades     # also collect trades
    python3 -m mantis.collector --stats      # show collection stats
"""
from __future__ import annotations

import json
import logging
import signal
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import load_config
from .polymarket_client import PolymarketClient
from .scanner import compute_total_q_min

logger = logging.getLogger("mantis.collector")

# Cross-platform path: use the directory where this module is located
_MODULE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = _MODULE_DIR / "data" / "snapshots.db"
DATA_API_BASE = "https://data-api.polymarket.com"


@dataclass
class VPINState:
    """Track VPIN computation state per market."""
    bucket_size: float = 30.0
    n_buckets: int = 30
    buckets: list = field(default_factory=list)
    current_buy: float = 0.0
    current_sell: float = 0.0
    bucket_vol: float = 0.0
    last_trade_id: str = ""

    def add_trade(self, size: float, side: str) -> float:
        """Add trade and return current VPIN."""
        if side.upper() == "BUY":
            self.current_buy += size
        else:
            self.current_sell += size
        self.bucket_vol += size

        if self.bucket_vol >= self.bucket_size:
            self.buckets.append((self.current_buy, self.current_sell))
            if len(self.buckets) > self.n_buckets:
                self.buckets = self.buckets[-self.n_buckets:]
            self.bucket_vol = 0
            self.current_buy = 0
            self.current_sell = 0

        return self.get_vpin()

    def get_vpin(self) -> float:
        if len(self.buckets) < 5:
            return 0.0
        total_imb = sum(abs(b - s) for b, s in self.buckets)
        total_vol = sum(b + s for b, s in self.buckets)
        return total_imb / total_vol if total_vol > 0 else 0.0


@dataclass
class OrderbookState:
    """Track orderbook changes for queue dynamics."""
    prev_bids: dict = field(default_factory=dict)  # price -> size
    prev_asks: dict = field(default_factory=dict)
    total_bid_adds: float = 0.0
    total_bid_removes: float = 0.0
    total_ask_adds: float = 0.0
    total_ask_removes: float = 0.0

    def update(self, bids: list, asks: list) -> dict:
        """Update state and return change metrics."""
        new_bids = {l.price: l.size for l in bids}
        new_asks = {l.price: l.size for l in asks}

        # Calculate changes
        bid_adds = 0.0
        bid_removes = 0.0
        for p, s in new_bids.items():
            if p not in self.prev_bids:
                bid_adds += s
            elif s > self.prev_bids[p]:
                bid_adds += s - self.prev_bids[p]
        for p, s in self.prev_bids.items():
            if p not in new_bids:
                bid_removes += s
            elif s > new_bids[p]:
                bid_removes += s - new_bids[p]

        ask_adds = 0.0
        ask_removes = 0.0
        for p, s in new_asks.items():
            if p not in self.prev_asks:
                ask_adds += s
            elif s > self.prev_asks[p]:
                ask_adds += s - self.prev_asks[p]
        for p, s in self.prev_asks.items():
            if p not in new_asks:
                ask_removes += s
            elif s > new_asks[p]:
                ask_removes += s - new_asks[p]

        self.total_bid_adds += bid_adds
        self.total_bid_removes += bid_removes
        self.total_ask_adds += ask_adds
        self.total_ask_removes += ask_removes

        self.prev_bids = new_bids
        self.prev_asks = new_asks

        return {
            "bid_adds": bid_adds,
            "bid_removes": bid_removes,
            "ask_adds": ask_adds,
            "ask_removes": ask_removes,
        }


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
            asks_json TEXT,
            -- Enhanced fields
            vpin REAL DEFAULT 0,
            trade_imbalance REAL DEFAULT 0,
            bid_adds REAL DEFAULT 0,
            bid_removes REAL DEFAULT 0,
            ask_adds REAL DEFAULT 0,
            ask_removes REAL DEFAULT 0
        )
    """)

    # Enhanced: Trade flow table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            ts_unix REAL NOT NULL,
            condition_id TEXT NOT NULL,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            mid_at_trade REAL
        )
    """)

    # Enhanced: VPIN time series
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vpin_series (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_unix REAL NOT NULL,
            condition_id TEXT NOT NULL,
            vpin REAL,
            buy_vol_30m REAL,
            sell_vol_30m REAL,
            imbalance REAL
        )
    """)

    # Enhanced: Q competition time series
    conn.execute("""
        CREATE TABLE IF NOT EXISTS q_competition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_unix REAL NOT NULL,
            condition_id TEXT NOT NULL,
            total_q REAL,
            n_makers_at_best REAL,
            depth_at_best_bid REAL,
            depth_at_best_ask REAL
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(condition_id, ts_unix)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(condition_id, ts_unix)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vpin_ts ON vpin_series(condition_id, ts_unix)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qcomp_ts ON q_competition(condition_id, ts_unix)")

    # Add new columns to existing tables (if they don't exist)
    for col, typ in [("vpin", "REAL"), ("trade_imbalance", "REAL"),
                     ("bid_adds", "REAL"), ("bid_removes", "REAL"),
                     ("ask_adds", "REAL"), ("ask_removes", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {typ} DEFAULT 0")
        except sqlite3.OperationalError:
            pass

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


def fetch_recent_trades(condition_id: str, http_client: httpx.Client) -> list[dict]:
    """Fetch recent trades from Polymarket public API."""
    try:
        resp = http_client.get(
            f"{DATA_API_BASE}/trades",
            params={"market": condition_id, "limit": "50"},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        trades = []
        for t in raw:
            trades.append({
                "id": f"{t.get('proxyWallet', '')}-{t.get('timestamp', '')}",
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", 0)),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("timestamp", 0)),
            })
        return trades
    except Exception as e:
        logger.debug(f"Failed to fetch trades: {e}")
        return []


def collect_trades(conn: sqlite3.Connection, condition_id: str, token_id: str,
                   trades: list[dict], vpin_state: VPINState, mid: float) -> tuple[int, float]:
    """Store new trades and update VPIN. Returns (new_trade_count, current_vpin)."""
    if not trades:
        return 0, vpin_state.get_vpin()

    # Get existing trade IDs
    trade_ids = [t["id"] for t in trades]
    placeholders = ",".join("?" * len(trade_ids))
    existing = conn.execute(
        f"SELECT trade_id FROM trades WHERE trade_id IN ({placeholders})",
        trade_ids,
    ).fetchall()
    existing_ids = {r[0] for r in existing}

    new_count = 0
    now = time.time()

    for t in trades:
        if t["id"] in existing_ids:
            continue

        # Store trade
        conn.execute("""
            INSERT OR IGNORE INTO trades
            (trade_id, ts_unix, condition_id, token_id, side, price, size, mid_at_trade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (t["id"], t.get("timestamp", now), condition_id, token_id,
              t["side"], t["price"], t["size"], mid))

        # Update VPIN
        vpin_state.add_trade(t["size"], t["side"])
        vpin_state.last_trade_id = t["id"]
        new_count += 1

    conn.commit()
    return new_count, vpin_state.get_vpin()


def collect_snapshot(client: PolymarketClient, conn: sqlite3.Connection, m: dict,
                     ob_state: OrderbookState = None,
                     vpin_state: VPINState = None,
                     http_client: httpx.Client = None,
                     collect_trades_flag: bool = False) -> tuple[bool, dict]:
    """Collect one orderbook snapshot with enhanced metrics. Returns (success, stats)."""
    stats = {"trades": 0, "vpin": 0.0}
    try:
        ob = client.fetch_orderbook(m["token_id"])
        if not ob.bids and not ob.asks:
            return False, stats

        now = datetime.now(timezone.utc)
        total_q = compute_total_q_min(ob, m["max_spread"])

        bid_depth_5 = sum(l.size for l in ob.bids[:5])
        ask_depth_5 = sum(l.size for l in ob.asks[:5])

        bids_json = json.dumps([{"p": l.price, "s": l.size} for l in ob.bids[:10]])
        asks_json = json.dumps([{"p": l.price, "s": l.size} for l in ob.asks[:10]])

        # Orderbook change tracking
        ob_changes = {"bid_adds": 0, "bid_removes": 0, "ask_adds": 0, "ask_removes": 0}
        if ob_state:
            ob_changes = ob_state.update(ob.bids, ob.asks)

        # Trade and VPIN tracking
        vpin = 0.0
        trade_imbalance = 0.0
        if collect_trades_flag and http_client and vpin_state:
            trades = fetch_recent_trades(m["condition_id"], http_client)
            new_trades, vpin = collect_trades(
                conn, m["condition_id"], m["token_id"], trades, vpin_state, ob.midpoint
            )
            stats["trades"] = new_trades
            stats["vpin"] = vpin

            # Calculate trade imbalance from recent trades
            recent = [t for t in trades if t.get("timestamp", 0) > now.timestamp() - 60]
            buy_vol = sum(t["size"] for t in recent if t["side"] == "BUY")
            sell_vol = sum(t["size"] for t in recent if t["side"] == "SELL")
            total_vol = buy_vol + sell_vol
            trade_imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0

        conn.execute("""
            INSERT INTO snapshots
            (ts, ts_unix, condition_id, token_id, best_bid, best_ask, midpoint,
             spread_cents, bid_depth_5, ask_depth_5, total_q, n_bid_levels, n_ask_levels,
             bids_json, asks_json, vpin, trade_imbalance,
             bid_adds, bid_removes, ask_adds, ask_removes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now.isoformat(), now.timestamp(), m["condition_id"], m["token_id"],
            ob.best_bid, ob.best_ask, ob.midpoint, ob.spread_cents,
            bid_depth_5, ask_depth_5, total_q,
            len(ob.bids), len(ob.asks), bids_json, asks_json,
            vpin, trade_imbalance,
            ob_changes["bid_adds"], ob_changes["bid_removes"],
            ob_changes["ask_adds"], ob_changes["ask_removes"],
        ))

        # Q competition tracking
        if ob.bids:
            best_bid_price = ob.bids[0].price
            depth_at_best_bid = sum(l.size for l in ob.bids if l.price == best_bid_price)
            n_makers_bid = sum(1 for l in ob.bids if l.price == best_bid_price)
        else:
            depth_at_best_bid = 0
            n_makers_bid = 0

        if ob.asks:
            best_ask_price = ob.asks[0].price
            depth_at_best_ask = sum(l.size for l in ob.asks if l.price == best_ask_price)
        else:
            depth_at_best_ask = 0

        conn.execute("""
            INSERT INTO q_competition
            (ts_unix, condition_id, total_q, n_makers_at_best, depth_at_best_bid, depth_at_best_ask)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (now.timestamp(), m["condition_id"], total_q, n_makers_bid,
              depth_at_best_bid, depth_at_best_ask))

        conn.commit()
        return True, stats
    except Exception as e:
        logger.debug(f"Snapshot failed for {m['condition_id'][:8]}: {e}")
        return False, stats


def run_collector(top_n: int = 10, interval: int = 30, config_path: str = "config.yaml",
                  collect_trades_flag: bool = True):
    """Main collector loop with enhanced data collection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    config = load_config(config_path)
    client = PolymarketClient()
    conn = init_db()
    http_client = httpx.Client(timeout=15) if collect_trades_flag else None

    # Per-market state tracking
    vpin_states: dict[str, VPINState] = {}
    ob_states: dict[str, OrderbookState] = {}

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
            # Initialize per-market state
            vpin_states[m["condition_id"]] = VPINState()
            ob_states[m["condition_id"]] = OrderbookState()
            logger.info(f"  #{i+1}: ${m['daily_rate']}/d | rpc={m['reward_per_capital']:.2f} | {m['question'][:50]}")

        # Collection loop
        cycle = 0
        total_snaps = 0
        total_trades = 0
        logger.info(f"Starting enhanced collection: {len(markets)} markets, every {interval}s")
        logger.info(f"Trade collection: {'ON' if collect_trades_flag else 'OFF'}")
        logger.info(f"Data saved to: {DB_PATH}")

        while running:
            cycle += 1
            t0 = time.time()
            ok = 0
            cycle_trades = 0
            avg_vpin = 0.0

            for m in markets:
                if not running:
                    break

                cid = m["condition_id"]
                success, stats = collect_snapshot(
                    client, conn, m,
                    ob_state=ob_states.get(cid),
                    vpin_state=vpin_states.get(cid),
                    http_client=http_client,
                    collect_trades_flag=collect_trades_flag,
                )
                if success:
                    ok += 1
                    cycle_trades += stats.get("trades", 0)
                    avg_vpin += stats.get("vpin", 0)

                time.sleep(0.05)  # rate limit

            total_snaps += ok
            total_trades += cycle_trades
            elapsed = time.time() - t0
            avg_vpin = avg_vpin / max(ok, 1)

            if cycle % 10 == 1:  # log every 10th cycle
                logger.info(
                    f"Cycle {cycle}: {ok}/{len(markets)} snaps in {elapsed:.1f}s | "
                    f"trades: {cycle_trades} | avg VPIN: {avg_vpin:.2f} | "
                    f"total: {total_snaps} snaps, {total_trades} trades"
                )

            # Wait for next cycle
            wait = max(0, interval - elapsed)
            if wait > 0 and running:
                time.sleep(wait)

            # Re-scan markets hourly
            if cycle % (3600 // interval) == 0:
                logger.info("Hourly rescan: checking for new markets...")
                new_markets = select_markets(client, config, top_n)
                new_cids = {m["condition_id"] for m in new_markets}
                old_cids = {m["condition_id"] for m in markets}
                added = new_cids - old_cids
                if added:
                    logger.info(f"  Added {len(added)} new markets")
                    for m in new_markets:
                        if m["condition_id"] in added:
                            save_market(conn, m)
                            vpin_states[m["condition_id"]] = VPINState()
                            ob_states[m["condition_id"]] = OrderbookState()
                    markets = new_markets

    finally:
        client.close()
        conn.close()
        if http_client:
            http_client.close()
        logger.info(f"Collector stopped. Total: {total_snaps} snapshots, {total_trades} trades")


def get_stats(db_path: Path = DB_PATH) -> str:
    """Get enhanced collection stats."""
    if not db_path.exists():
        return "No data collected yet."

    conn = sqlite3.connect(str(db_path))
    lines = []

    # Overall stats
    row = conn.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM snapshots").fetchone()
    total, first_ts, last_ts = row
    lines.append("=" * 60)
    lines.append("ENHANCED COLLECTOR STATISTICS")
    lines.append("=" * 60)
    lines.append(f"Total snapshots: {total}")
    lines.append(f"First: {first_ts}")
    lines.append(f"Last:  {last_ts}")
    lines.append("")

    # Trade stats
    try:
        trade_row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        trade_count = trade_row[0] if trade_row else 0
        lines.append(f"Total trades collected: {trade_count}")

        # Trade stats per market
        if trade_count > 0:
            trade_stats = conn.execute("""
                SELECT condition_id, COUNT(*), SUM(size), AVG(price)
                FROM trades GROUP BY condition_id
            """).fetchall()
            lines.append(f"  Markets with trade data: {len(trade_stats)}")
    except sqlite3.OperationalError:
        lines.append("Trade collection: not enabled")

    lines.append("")

    # VPIN stats
    try:
        vpin_row = conn.execute("""
            SELECT AVG(vpin), MIN(vpin), MAX(vpin)
            FROM snapshots WHERE vpin > 0
        """).fetchone()
        if vpin_row and vpin_row[0]:
            lines.append(f"VPIN stats:")
            lines.append(f"  Average: {vpin_row[0]:.3f}")
            lines.append(f"  Min/Max: {vpin_row[1]:.3f} / {vpin_row[2]:.3f}")

            # High VPIN percentage
            high_vpin = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE vpin > 0.35"
            ).fetchone()[0]
            total_with_vpin = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE vpin > 0"
            ).fetchone()[0]
            if total_with_vpin > 0:
                lines.append(f"  High VPIN (>0.35): {high_vpin / total_with_vpin * 100:.1f}%")
    except sqlite3.OperationalError:
        pass

    lines.append("")

    # Orderbook dynamics
    try:
        ob_row = conn.execute("""
            SELECT AVG(bid_adds), AVG(bid_removes), AVG(ask_adds), AVG(ask_removes)
            FROM snapshots WHERE bid_adds > 0 OR bid_removes > 0
        """).fetchone()
        if ob_row and ob_row[0]:
            lines.append("Orderbook dynamics (avg per snapshot):")
            lines.append(f"  Bid adds: {ob_row[0]:.1f} | Bid removes: {ob_row[1]:.1f}")
            lines.append(f"  Ask adds: {ob_row[2]:.1f} | Ask removes: {ob_row[3]:.1f}")
    except sqlite3.OperationalError:
        pass

    lines.append("")

    # Q competition stats
    try:
        q_row = conn.execute("""
            SELECT AVG(total_q), AVG(depth_at_best_bid), AVG(depth_at_best_ask)
            FROM q_competition
        """).fetchone()
        if q_row and q_row[0]:
            lines.append("Q Competition:")
            lines.append(f"  Avg total Q: {q_row[0]:.1f}")
            lines.append(f"  Avg depth at best bid: {q_row[1]:.1f}")
            lines.append(f"  Avg depth at best ask: {q_row[2]:.1f}")
    except sqlite3.OperationalError:
        pass

    lines.append("")

    # Per market breakdown
    lines.append("--- Per Market ---")
    rows = conn.execute("""
        SELECT m.question, s.condition_id, COUNT(*) as cnt,
               MIN(s.midpoint) as min_mid, MAX(s.midpoint) as max_mid,
               AVG(s.spread_cents) as avg_spread, AVG(s.total_q) as avg_q,
               AVG(s.vpin) as avg_vpin
        FROM snapshots s
        JOIN markets m ON s.condition_id = m.condition_id
        GROUP BY s.condition_id
        ORDER BY cnt DESC
    """).fetchall()

    lines.append(f"Markets tracked: {len(rows)}")
    lines.append("")
    for row in rows:
        q, cid, cnt, min_mid, max_mid, avg_spread, avg_q, avg_vpin = row
        avg_vpin = avg_vpin or 0
        lines.append(f"  {q[:50]}")
        lines.append(f"    Snaps: {cnt} | Mid: {min_mid:.3f}-{max_mid:.3f} | "
                     f"Spread: {avg_spread:.1f}c | Q: {avg_q:.0f} | VPIN: {avg_vpin:.2f}")

    conn.close()
    return "\n".join(lines)


def export_calibration_data(db_path: Path = DB_PATH) -> dict:
    """Export calibration parameters from collected data.

    Returns a dict suitable for plugging into realistic_backtest.py.
    """
    if not db_path.exists():
        return {"error": "No data collected yet."}

    conn = sqlite3.connect(str(db_path))
    result = {}

    # Queue depth stats
    row = conn.execute("""
        SELECT AVG(bid_depth_5), AVG(ask_depth_5), AVG(total_q)
        FROM snapshots
    """).fetchone()
    if row:
        result["avg_bid_depth_5"] = row[0] or 0
        result["avg_ask_depth_5"] = row[1] or 0
        result["avg_total_q"] = row[2] or 0

    # VPIN stats
    try:
        row = conn.execute("""
            SELECT AVG(vpin), STDEV(vpin) FROM snapshots WHERE vpin > 0
        """).fetchone()
        if row and row[0]:
            result["avg_vpin"] = row[0]
    except:
        pass

    # Trade stats
    try:
        row = conn.execute("""
            SELECT COUNT(*), AVG(size), SUM(size)
            FROM trades
        """).fetchone()
        if row:
            result["total_trades"] = row[0] or 0
            result["avg_trade_size"] = row[1] or 0

        # Trades per hour
        time_row = conn.execute("""
            SELECT MIN(ts_unix), MAX(ts_unix) FROM trades
        """).fetchone()
        if time_row and time_row[0] and time_row[1]:
            hours = (time_row[1] - time_row[0]) / 3600
            if hours > 0:
                result["avg_trades_per_hour"] = result["total_trades"] / hours
    except:
        pass

    # Orderbook dynamics
    try:
        row = conn.execute("""
            SELECT AVG(bid_adds + ask_adds), AVG(bid_removes + ask_removes)
            FROM snapshots WHERE bid_adds > 0 OR ask_adds > 0
        """).fetchone()
        if row:
            result["avg_queue_additions"] = row[0] or 0
            result["avg_queue_removals"] = row[1] or 0
    except:
        pass

    conn.close()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mantis Enhanced Orderbook Collector")
    parser.add_argument("--top", type=int, default=10, help="Number of markets to track")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between snapshots")
    parser.add_argument("--stats", action="store_true", help="Show collection stats and exit")
    parser.add_argument("--trades", action="store_true", help="Also collect trade data (default: True)")
    parser.add_argument("--no-trades", action="store_true", help="Disable trade collection")
    parser.add_argument("--calibrate", action="store_true", help="Export calibration parameters")
    args = parser.parse_args()

    if args.stats:
        print(get_stats())
    elif args.calibrate:
        import json as _json
        params = export_calibration_data()
        print(_json.dumps(params, indent=2))
    else:
        import os
        os.chdir(str(_MODULE_DIR))
        collect_trades = not args.no_trades  # Default to True
        run_collector(
            top_n=args.top,
            interval=args.interval,
            collect_trades_flag=collect_trades,
        )
