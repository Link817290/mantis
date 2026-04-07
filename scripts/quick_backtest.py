"""Quick backtest - fetch live data from Polymarket and simulate.

No pre-collected data needed. Fetches current orderbooks and historical trades
directly from Polymarket APIs.

Usage:
    python scripts/quick_backtest.py              # backtest top 5 markets
    python scripts/quick_backtest.py --top 10    # backtest top 10 markets
"""
import json
import math
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

# API endpoints
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

# Calibrated parameters
FILL_PROB = 0.278
ADVERSE_RATE = 0.06
ADVERSE_RATE_HIGH_VPIN = 0.18
ADVERSE_RATE_LOW_VPIN = 0.04
ADVERSE_MOVE = 0.0155
QUEUE_DEPTH = 130
GAS_PER_TX = 0.002
COOLDOWN_SEC = 60


def q_score(size, dist_c, max_spread_c):
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


@dataclass
class VPINTracker:
    buckets: list = field(default_factory=list)
    current_buy: float = 0.0
    current_sell: float = 0.0
    bucket_vol: float = 0.0

    def add_trade(self, size: float, side: str):
        if side == "BUY":
            self.current_buy += size
        else:
            self.current_sell += size
        self.bucket_vol += size
        if self.bucket_vol >= 30:
            self.buckets.append((self.current_buy, self.current_sell))
            if len(self.buckets) > 30:
                self.buckets = self.buckets[-30:]
            self.bucket_vol = 0
            self.current_buy = 0
            self.current_sell = 0

    def get_vpin(self) -> float:
        if len(self.buckets) < 5:
            return 0.0
        total_imb = sum(abs(b - s) for b, s in self.buckets)
        total_vol = sum(b + s for b, s in self.buckets)
        return total_imb / total_vol if total_vol > 0 else 0.0


@dataclass
class Inventory:
    yes_qty: float = 0.0
    yes_avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def buy_yes(self, qty, price):
        if self.yes_qty + qty > 0:
            total_cost = self.yes_avg_cost * self.yes_qty + price * qty
            self.yes_qty += qty
            self.yes_avg_cost = total_cost / self.yes_qty
        else:
            self.yes_qty += qty
            self.yes_avg_cost = price

    def sell_yes(self, qty, price):
        if self.yes_qty > 0:
            close_qty = min(qty, self.yes_qty)
            self.realized_pnl += close_qty * (price - self.yes_avg_cost)
            self.yes_qty -= close_qty

    def mtm_pnl(self, mid):
        if self.yes_qty > 0:
            return self.yes_qty * (mid - self.yes_avg_cost)
        return 0.0

    def liquidate(self, mid):
        liq_pnl = 0.0
        if self.yes_qty > 0:
            liq_pnl = self.yes_qty * (mid - self.yes_avg_cost)
            self.yes_qty = 0
        self.realized_pnl += liq_pnl
        return liq_pnl


def fetch_sampling_markets(http: httpx.Client) -> list:
    """Fetch markets with rewards from CLOB API."""
    try:
        resp = http.get(f"{CLOB_BASE}/sampling-markets", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # API returns {"data": [...], "next_cursor": ..., "limit": ..., "count": ...}
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data or []
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def fetch_orderbook(http: httpx.Client, token_id: str) -> dict:
    """Fetch orderbook for a token."""
    try:
        resp = http.get(f"{CLOB_BASE}/book", params={"token_id": token_id}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"bids": [], "asks": []}


def fetch_trades(http: httpx.Client, condition_id: str, limit: int = 500) -> list:
    """Fetch historical trades for a market."""
    try:
        resp = http.get(
            f"{DATA_BASE}/trades",
            params={"market": condition_id, "limit": str(limit)},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        trades = []
        for t in raw:
            trades.append({
                "side": t.get("side", "").upper(),
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", 0)),
                "timestamp": int(t.get("timestamp", 0)),
            })
        trades.sort(key=lambda x: x["timestamp"])
        return trades
    except Exception as e:
        print(f"  Error fetching trades: {e}")
        return []


def select_markets(http: httpx.Client, top_n: int = 5) -> list:
    """Select top markets by reward potential."""
    print("Fetching markets from Polymarket...")
    all_markets = fetch_sampling_markets(http)
    print(f"  Found {len(all_markets)} sampling markets")

    candidates = []
    for m in all_markets:
        try:
            if not m.get("active"):
                continue
            rewards = m.get("rewards") or {}
            rates = rewards.get("rates") or []
            daily_rate = rates[0].get("rewards_daily_rate", 0) if rates else 0
            min_size = rewards.get("min_size", 0)
            max_spread = rewards.get("max_spread", 0)

            # Relaxed filters for more markets
            if daily_rate < 0.001 or min_size > 200 or max_spread < 1:
                continue

            tokens = m.get("tokens") or []
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            if not yes_token:
                continue

            price = float(yes_token.get("price", 0.5))
            if not (0.05 <= price <= 0.95):
                continue

            candidates.append({
                "condition_id": m.get("condition_id"),
                "question": (m.get("question") or "")[:60],
                "token_id": yes_token.get("token_id"),
                "daily_rate": daily_rate,
                "min_size": min_size,
                "max_spread": max_spread,
                "price": price,
            })
        except Exception as e:
            continue

    candidates.sort(key=lambda x: -x["daily_rate"])
    print(f"  {len(candidates)} markets pass filters")
    return candidates[:top_n * 2]  # fetch more, filter later


def simulate_market(trades: list, daily_rate: float, max_spread: float,
                    min_size: int, spread_pct: float = 0.15,
                    use_vpin: bool = True) -> dict:
    """Simulate trading on historical trades."""
    if len(trades) < 20:
        return None

    t_start = trades[0]["timestamp"]
    t_end = trades[-1]["timestamp"]
    duration_sec = t_end - t_start
    n_days = max(duration_sec / 86400, 0.001)

    inv = Inventory()
    vpin = VPINTracker() if use_vpin else None
    ema_mid = trades[0]["price"]

    fills = 0
    toxic_fills = 0
    total_gas = 0.0
    reprice_count = 0

    bid_live = True
    ask_live = True
    bid_cooldown_until = 0
    ask_cooldown_until = 0

    bid_on_book_sec = 0.0
    ask_on_book_sec = 0.0
    last_ts = t_start
    last_reprice_mid = ema_mid

    for trade in trades:
        ts = trade["timestamp"]
        price = trade["price"]
        size = trade["size"]
        side = trade["side"]

        if price <= 0.01 or price >= 0.99:
            continue

        dt = ts - last_ts
        if bid_live:
            bid_on_book_sec += dt
        if ask_live:
            ask_on_book_sec += dt

        # Update VPIN
        if vpin:
            vpin.add_trade(size, side)
        current_vpin = vpin.get_vpin() if vpin else 0

        # Cooldown check
        if not bid_live and ts >= bid_cooldown_until:
            bid_live = True
        if not ask_live and ts >= ask_cooldown_until:
            ask_live = True

        # Dynamic spread based on VPIN
        eff_pct = spread_pct
        if use_vpin and current_vpin > 0.35:
            eff_pct = min(0.40, spread_pct * 1.5)

        dist_c = max_spread * eff_pct
        half = max(dist_c / 100, 0.001)

        # Reprice check
        if abs(ema_mid - last_reprice_mid) >= 0.005:
            last_reprice_mid = ema_mid
            reprice_count += 1
            total_gas += 4 * GAS_PER_TX

        our_bid = last_reprice_mid - half
        our_ask = last_reprice_mid + half

        # Fill check
        fill_prob = min_size / (min_size + QUEUE_DEPTH)

        if side == "SELL" and price <= our_bid + 0.005 and bid_live:
            if random.random() < fill_prob:
                inv.buy_yes(min_size, our_bid)
                fills += 1
                bid_live = False
                bid_cooldown_until = ts + COOLDOWN_SEC

                # Adverse selection based on VPIN
                adverse_rate = ADVERSE_RATE_HIGH_VPIN if current_vpin > 0.3 else ADVERSE_RATE_LOW_VPIN
                if current_vpin > 0.35:
                    toxic_fills += 1

        elif side == "BUY" and price >= our_ask - 0.005 and ask_live:
            if random.random() < fill_prob:
                inv.sell_yes(min_size, our_ask)
                fills += 1
                ask_live = False
                ask_cooldown_until = ts + COOLDOWN_SEC

                adverse_rate = ADVERSE_RATE_HIGH_VPIN if current_vpin > 0.3 else ADVERSE_RATE_LOW_VPIN
                if current_vpin > 0.35:
                    toxic_fills += 1

        ema_mid = 0.3 * price + 0.7 * ema_mid
        last_ts = ts

    # Liquidate
    final_mid = ema_mid
    liq_pnl = inv.liquidate(final_mid)

    # Reward calculation
    avg_on_book_sec = min(bid_on_book_sec, ask_on_book_sec)
    on_book_ratio = avg_on_book_sec / max(duration_sec, 1)

    dist_c_reward = max_spread * spread_pct
    our_q = q_score(min_size, dist_c_reward, max_spread)
    total_q = 100.0  # estimate
    our_share = our_q / (total_q + our_q)
    total_reward = daily_rate * n_days * our_share * on_book_ratio

    net = total_reward + inv.realized_pnl + liq_pnl - total_gas

    return {
        "fills": fills,
        "toxic_fills": toxic_fills,
        "reward": total_reward,
        "realized_pnl": inv.realized_pnl,
        "liq_pnl": liq_pnl,
        "gas": total_gas,
        "net": net,
        "n_days": n_days,
        "n_trades": len(trades),
        "on_book_ratio": on_book_ratio,
        "reprices": reprice_count,
    }


def run_backtest(top_n: int = 5):
    """Run backtest on top markets."""
    http = httpx.Client(timeout=30)

    try:
        markets = select_markets(http, top_n)
        print(f"\nSelected {len(markets)} candidate markets\n")

        results = []
        for i, m in enumerate(markets):
            if len(results) >= top_n:
                break

            print(f"[{i+1}/{len(markets)}] {m['question']}")
            print(f"  Daily rate: ${m['daily_rate']:.2f}, min_size: {m['min_size']}")

            trades = fetch_trades(http, m["condition_id"])
            if len(trades) < 20:
                print(f"  Skipped: only {len(trades)} trades")
                continue

            # Run simulation
            res = simulate_market(
                trades, m["daily_rate"], m["max_spread"], m["min_size"],
                spread_pct=0.15, use_vpin=True,
            )
            if res:
                res["question"] = m["question"]
                res["daily_rate"] = m["daily_rate"]
                results.append(res)
                print(f"  {res['n_trades']} trades over {res['n_days']:.1f} days")
                print(f"  Fills: {res['fills']} (toxic: {res['toxic_fills']})")
                print(f"  NET: ${res['net']:.2f} (reward: ${res['reward']:.2f}, fill P&L: ${res['realized_pnl']+res['liq_pnl']:.2f})")

            time.sleep(0.2)  # rate limit

        # Summary
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        if not results:
            print("No markets successfully backtested.")
            return

        total_net = sum(r["net"] for r in results)
        total_reward = sum(r["reward"] for r in results)
        total_fills = sum(r["fills"] for r in results)
        total_toxic = sum(r["toxic_fills"] for r in results)
        total_gas = sum(r["gas"] for r in results)

        print(f"Markets tested:     {len(results)}")
        print(f"Total NET profit:   ${total_net:.2f}")
        print(f"  Reward earned:    ${total_reward:.2f}")
        print(f"  Fill count:       {total_fills} (toxic: {total_toxic})")
        print(f"  Gas cost:         ${total_gas:.2f}")
        print()

        print("Per-market breakdown:")
        for r in sorted(results, key=lambda x: -x["net"]):
            toxic_pct = r["toxic_fills"] / max(r["fills"], 1) * 100
            print(f"  {r['question'][:45]}")
            print(f"    NET: ${r['net']:>7.2f} | fills: {r['fills']:>3} | toxic: {toxic_pct:.0f}% | days: {r['n_days']:.1f}")

        # Daily projection
        total_days = sum(r["n_days"] for r in results) / len(results)
        daily_net = total_net / max(total_days, 1)
        print()
        print(f"Average period:     {total_days:.1f} days")
        print(f"Projected daily:    ${daily_net:.2f}/day")
        print(f"Projected monthly:  ${daily_net * 30:.2f}/month")

    finally:
        http.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick Polymarket Backtest")
    parser.add_argument("--top", type=int, default=5, help="Number of markets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    run_backtest(top_n=args.top)
