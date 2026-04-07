"""Bulk data fetcher: collect trades + orderbooks for many markets.

Fetches from:
- CLOB API: sampling-markets (reward info), orderbooks
- Gamma API: market metadata (volume, liquidity, spread, price)
- Data API: historical trades

Stores everything in SQLite for backtesting.
"""
import sys
sys.path.insert(0, "/workspace/mantis")

import json
import math
import sqlite3
import time
from pathlib import Path

import httpx

DB_PATH = Path("/workspace/mantis/data/backtest.db")
CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

HEADERS = {"Accept": "application/json", "User-Agent": "Mantis/0.2"}


def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT,
            token_id_yes TEXT,
            token_id_no TEXT,
            daily_rate REAL,
            min_size INTEGER,
            max_spread REAL,
            best_bid REAL,
            best_ask REAL,
            spread REAL,
            volume_24h REAL,
            volume_1w REAL,
            volume_1m REAL,
            liquidity REAL,
            gm_reward_per_100 REAL,
            volatility_sum REAL,
            profitability_score REAL,
            fetched_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT NOT NULL,
            side TEXT,
            price REAL,
            size REAL,
            timestamp INTEGER,
            asset TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT NOT NULL,
            token_id TEXT,
            ts REAL,
            best_bid REAL,
            best_ask REAL,
            midpoint REAL,
            spread_cents REAL,
            total_q REAL,
            bids_json TEXT,
            asks_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_cid ON trades(condition_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ob_cid ON orderbook_snapshots(condition_id, ts)")
    conn.commit()
    return conn


def q_score(size, dist_c, max_spread_c):
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


def fetch_all_sampling_markets(client: httpx.Client) -> list[dict]:
    """Fetch all reward-eligible markets from CLOB."""
    all_markets = []
    cursor = ""
    while True:
        params = {}
        if cursor:
            params["next_cursor"] = cursor
        r = client.get(f"{CLOB_BASE}/sampling-markets", params=params)
        r.raise_for_status()
        data = r.json()
        markets = data.get("data", [])
        all_markets.extend(markets)
        cursor = data.get("next_cursor", "")
        if not cursor or cursor == "LTE=" or not markets:
            break
        time.sleep(0.1)
    print(f"Fetched {len(all_markets)} sampling markets")
    return all_markets


def fetch_gamma_markets(client: httpx.Client, limit=500) -> dict:
    """Fetch market metadata from gamma API. Returns dict keyed by condition_id."""
    result = {}
    offset = 0
    while offset < 3000:
        r = client.get(f"{GAMMA_BASE}/markets", params={
            "limit": min(limit, 100), "offset": offset,
            "active": "true", "closed": "false",
        })
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        for m in data:
            cid = m.get("conditionId", "")
            if cid:
                result[cid] = m
        offset += len(data)
        if len(data) < 100:
            break
        time.sleep(0.1)
    print(f"Fetched {len(result)} gamma markets")
    return result


def compute_gm_reward(daily_rate, min_size, max_spread, best_bid, best_ask, total_q_existing):
    """Compute geometric mean reward per $100 capital deployed.

    This is defiance_cr's key market selection metric.
    """
    if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
        return 0.0

    mid = (best_bid + best_ask) / 2
    if mid <= 0.05 or mid >= 0.95:
        return 0.0

    # Order distance = 15% of max_spread (defiance's optimal)
    dist_pct = 0.15
    dist_c = max_spread * dist_pct

    # Our Q score at this distance
    our_q = q_score(min_size, dist_c, max_spread)

    # Estimate total Q (existing + ours)
    total_q = max(total_q_existing, 1.0) + our_q
    our_share = our_q / total_q

    # Capital needed: buy side + sell side
    cap_bid = min_size * mid
    cap_ask = min_size * (1 - mid)

    # Reward for each side
    reward_bid = daily_rate * our_share
    reward_ask = daily_rate * our_share

    # Reward per $100 for each side
    rpc_bid = (reward_bid / cap_bid * 100) if cap_bid > 0 else 0
    rpc_ask = (reward_ask / cap_ask * 100) if cap_ask > 0 else 0

    # Geometric mean
    if rpc_bid > 0 and rpc_ask > 0:
        return math.sqrt(rpc_bid * rpc_ask)
    return 0.0


def compute_volatility_from_trades(trades: list[dict]) -> dict:
    """Compute volatility metrics from trade history."""
    if len(trades) < 10:
        return {"vol_24h": 999, "vol_7d": 999, "vol_14d": 999, "vol_sum": 999}

    now = trades[-1]["timestamp"] if trades else time.time()

    def vol_for_period(seconds):
        cutoff = now - seconds
        prices = [t["price"] for t in trades if t["timestamp"] >= cutoff and 0 < t["price"] < 1]
        if len(prices) < 5:
            return 10.0  # high default
        returns = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        if not returns:
            return 0.0
        import statistics
        return statistics.stdev(returns) * 100 if len(returns) > 1 else 0.0

    v24 = vol_for_period(86400)
    v7d = vol_for_period(7 * 86400)
    v14d = vol_for_period(14 * 86400)

    return {"vol_24h": v24, "vol_7d": v7d, "vol_14d": v14d, "vol_sum": v24 + v7d + v14d}


def fetch_trades_for_market(client: httpx.Client, condition_id: str, max_trades=2000) -> list[dict]:
    """Fetch historical trades for a market from data API."""
    all_trades = []
    offset = 0
    while len(all_trades) < max_trades:
        try:
            r = client.get(f"{DATA_BASE}/trades", params={
                "market": condition_id, "limit": 100, "offset": offset,
            }, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
            all_trades.extend(data)
            offset += len(data)
            if len(data) < 100:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"  Trade fetch error at offset {offset}: {e}")
            break
    return all_trades


def fetch_orderbook(client: httpx.Client, token_id: str) -> dict:
    """Fetch current orderbook."""
    try:
        r = client.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"bids": [], "asks": []}


def compute_total_q(ob: dict, max_spread: float) -> float:
    """Compute total Q from orderbook."""
    total = 0.0
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    if not bids or not asks:
        return 0.0

    best_bid = float(bids[0]["price"]) if bids else 0
    best_ask = float(asks[0]["price"]) if asks else 1
    mid = (best_bid + best_ask) / 2

    for level in bids:
        price = float(level["price"])
        size = float(level["size"])
        dist_c = abs(mid - price) * 100
        total += q_score(size, dist_c, max_spread)

    for level in asks:
        price = float(level["price"])
        size = float(level["size"])
        dist_c = abs(price - mid) * 100
        total += q_score(size, dist_c, max_spread)

    return total


def run_fetcher():
    client = httpx.Client(headers=HEADERS, timeout=15)
    conn = init_db()

    # Step 1: Get all sampling markets
    print("=== Step 1: Fetching sampling markets ===")
    sampling = fetch_all_sampling_markets(client)

    # Step 2: Get gamma metadata
    print("\n=== Step 2: Fetching gamma metadata ===")
    gamma = fetch_gamma_markets(client)

    # Step 3: Filter and rank markets
    print("\n=== Step 3: Filtering and ranking markets ===")
    candidates = []

    for m in sampling:
        cid = m["condition_id"]
        rewards = m.get("rewards", {})
        rates = rewards.get("rates", [{}])
        daily_rate = rates[0].get("rewards_daily_rate", 0) if rates else 0

        if daily_rate < 0.5:
            continue
        if not m.get("active"):
            continue

        min_size = rewards.get("min_size", 20)
        max_spread = rewards.get("max_spread", 3.5)

        tokens = m.get("tokens", [])
        yes_token = None
        no_token = None
        for t in tokens:
            if t.get("outcome") == "Yes":
                yes_token = t
            elif t.get("outcome") == "No":
                no_token = t

        if not yes_token:
            continue

        price = yes_token.get("price", 0)
        if not (0.10 <= price <= 0.90):
            continue

        # Get gamma data if available
        gm = gamma.get(cid, {})
        best_bid = float(gm.get("bestBid") or price - 0.005)
        best_ask = float(gm.get("bestAsk") or price + 0.005)
        spread = float(gm.get("spread") or 0.01)
        vol_24h = float(gm.get("volume24hrClob") or 0)
        vol_1w = float(gm.get("volume1wkClob") or 0)
        vol_1m = float(gm.get("volume1moClob") or 0)
        liquidity = float(gm.get("liquidityClob") or 0)

        # Filter: spread must be reasonable
        if spread > 0.10:
            continue

        candidates.append({
            "condition_id": cid,
            "question": m.get("question", ""),
            "token_id_yes": yes_token["token_id"],
            "token_id_no": no_token["token_id"] if no_token else "",
            "daily_rate": daily_rate,
            "min_size": min_size,
            "max_spread": max_spread,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "vol_24h": vol_24h,
            "vol_1w": vol_1w,
            "vol_1m": vol_1m,
            "liquidity": liquidity,
        })

    print(f"  {len(candidates)} candidates after basic filters")

    # Step 4: Fetch orderbooks for top candidates (by daily_rate) to compute Q
    print("\n=== Step 4: Fetching orderbooks for top candidates ===")
    candidates.sort(key=lambda c: -c["daily_rate"])
    top_n = min(80, len(candidates))

    for i, c in enumerate(candidates[:top_n]):
        ob = fetch_orderbook(client, c["token_id_yes"])
        total_q = compute_total_q(ob, c["max_spread"])

        gm_rpc = compute_gm_reward(
            c["daily_rate"], c["min_size"], c["max_spread"],
            c["best_bid"], c["best_ask"], total_q,
        )
        c["total_q"] = total_q
        c["gm_reward_per_100"] = gm_rpc

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{top_n} orderbooks")
        time.sleep(0.05)

    # Mark remaining without orderbook data
    for c in candidates[top_n:]:
        c["total_q"] = 0
        c["gm_reward_per_100"] = 0

    # Step 5: Select markets with gm_reward >= 0.5% and fetch trades
    selected = [c for c in candidates[:top_n] if c.get("gm_reward_per_100", 0) >= 0.5]
    selected.sort(key=lambda c: -c["gm_reward_per_100"])

    print(f"\n=== Step 5: {len(selected)} markets with gm_reward >= 0.5% ===")
    for c in selected[:20]:
        print(f"  ${c['daily_rate']:5.1f}/d | gm_rpc={c['gm_reward_per_100']:.2f}% | Q={c['total_q']:.0f} | {c['question'][:50]}")

    # Step 6: Fetch trades for selected markets
    print(f"\n=== Step 6: Fetching trades for {len(selected)} markets ===")

    for i, c in enumerate(selected):
        cid = c["condition_id"]

        # Check if we already have trades
        existing = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE condition_id = ?", (cid,)
        ).fetchone()[0]

        if existing > 500:
            print(f"  [{i+1}/{len(selected)}] Skip {c['question'][:40]} (already {existing} trades)")
            continue

        print(f"  [{i+1}/{len(selected)}] Fetching trades: {c['question'][:45]}...")
        trades = fetch_trades_for_market(client, cid, max_trades=2000)

        if trades:
            # Compute volatility
            trade_dicts = [{"price": float(t.get("price", 0)), "timestamp": int(t.get("timestamp", 0))}
                          for t in trades if t.get("price")]
            vol = compute_volatility_from_trades(trade_dicts)
            c["vol_sum"] = vol["vol_sum"]
            c["profitability_score"] = c["gm_reward_per_100"] / (vol["vol_sum"] + 1) if vol["vol_sum"] < 999 else 0

            # Insert trades
            for t in trades:
                conn.execute(
                    "INSERT INTO trades (condition_id, side, price, size, timestamp, asset) VALUES (?, ?, ?, ?, ?, ?)",
                    (cid, t.get("side", ""), float(t.get("price", 0)), float(t.get("size", 0)),
                     int(t.get("timestamp", 0)), t.get("asset", "")),
                )
            conn.commit()
            print(f"    Got {len(trades)} trades, vol_sum={vol['vol_sum']:.1f}")
        else:
            c["vol_sum"] = 999
            c["profitability_score"] = 0
            print(f"    No trades found")

        time.sleep(0.1)

    # Step 7: Fetch orderbook snapshots for selected markets
    print(f"\n=== Step 7: Saving orderbook snapshots ===")
    for c in selected:
        ob = fetch_orderbook(client, c["token_id_yes"])
        if not ob.get("bids") and not ob.get("asks"):
            continue

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        mid = (best_bid + best_ask) / 2
        spread_c = (best_ask - best_bid) * 100
        total_q = compute_total_q(ob, c["max_spread"])

        conn.execute(
            "INSERT INTO orderbook_snapshots (condition_id, token_id, ts, best_bid, best_ask, midpoint, spread_cents, total_q, bids_json, asks_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c["condition_id"], c["token_id_yes"], time.time(), best_bid, best_ask, mid, spread_c, total_q,
             json.dumps([{"p": b["price"], "s": b["size"]} for b in bids[:10]]),
             json.dumps([{"p": a["price"], "s": a["size"]} for a in asks[:10]])),
        )
        time.sleep(0.05)
    conn.commit()

    # Step 8: Save market metadata
    print(f"\n=== Step 8: Saving market metadata ===")
    for c in selected:
        conn.execute("""
            INSERT OR REPLACE INTO markets
            (condition_id, question, token_id_yes, token_id_no, daily_rate, min_size, max_spread,
             best_bid, best_ask, spread, volume_24h, volume_1w, volume_1m, liquidity,
             gm_reward_per_100, volatility_sum, profitability_score, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (c["condition_id"], c["question"], c["token_id_yes"], c.get("token_id_no", ""),
              c["daily_rate"], c["min_size"], c["max_spread"],
              c["best_bid"], c["best_ask"], c["spread"],
              c["vol_24h"], c["vol_1w"], c["vol_1m"], c["liquidity"],
              c.get("gm_reward_per_100", 0), c.get("vol_sum", 999), c.get("profitability_score", 0),
              time.time()))
    conn.commit()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_ob = conn.execute("SELECT COUNT(*) FROM orderbook_snapshots").fetchone()[0]

    print(f"Markets stored:     {total_markets}")
    print(f"Trades stored:      {total_trades}")
    print(f"OB snapshots:       {total_ob}")

    # Top 15 by profitability score
    print(f"\nTOP 15 MARKETS BY PROFITABILITY SCORE:")
    print(f"{'Rate':>6} {'gm_rpc':>7} {'vol_sum':>8} {'score':>7} {'trades':>7}  Question")
    print("-" * 85)

    rows = conn.execute("""
        SELECT m.daily_rate, m.gm_reward_per_100, m.volatility_sum, m.profitability_score,
               m.question, m.condition_id,
               (SELECT COUNT(*) FROM trades t WHERE t.condition_id = m.condition_id) as trade_count
        FROM markets m
        ORDER BY m.profitability_score DESC
        LIMIT 15
    """).fetchall()

    for dr, gm, vs, ps, q, cid, tc in rows:
        print(f"${dr:5.1f} {gm:6.2f}% {vs:7.1f} {ps:7.3f} {tc:7d}  {q[:45]}")

    conn.close()
    client.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    run_fetcher()
