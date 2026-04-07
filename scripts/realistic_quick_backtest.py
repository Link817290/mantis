"""Realistic backtest with fixed capital constraint.

Models real-world issues:
1. Fixed capital - can't trade markets that need more than available
2. Single-side exposure - when price is extreme, may only afford one side
3. Inventory risk - being filled means holding position, need to unwind
4. Adverse selection - fills happen when price moves against you
5. Queue position - you're at the back of the queue, may not get filled

Usage:
    python scripts/realistic_quick_backtest.py --capital 100
    python scripts/realistic_quick_backtest.py --capital 500 --top 10
"""
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import httpx

CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

# Calibrated from real data
QUEUE_DEPTH_MULTIPLIER = 3.0  # 队列深度 = 平均交易size × 3
GAS_PER_TX = 0.002
COOLDOWN_SEC = 60


@dataclass
class Position:
    """Track position and P&L for a single market."""
    yes_qty: float = 0.0
    yes_avg_cost: float = 0.0
    no_qty: float = 0.0
    no_avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def buy_yes(self, qty: float, price: float):
        """Buy YES shares."""
        cost = qty * price
        if self.yes_qty > 0:
            total_cost = self.yes_avg_cost * self.yes_qty + cost
            self.yes_qty += qty
            self.yes_avg_cost = total_cost / self.yes_qty
        else:
            self.yes_qty = qty
            self.yes_avg_cost = price
        return cost

    def sell_yes(self, qty: float, price: float):
        """Sell YES shares (close long or open short via NO)."""
        if self.yes_qty >= qty:
            # Close long
            pnl = qty * (price - self.yes_avg_cost)
            self.realized_pnl += pnl
            self.yes_qty -= qty
            return qty * price  # revenue
        elif self.yes_qty > 0:
            # Partial close + open NO
            close_qty = self.yes_qty
            pnl = close_qty * (price - self.yes_avg_cost)
            self.realized_pnl += pnl
            self.yes_qty = 0
            # Open NO position for remainder
            remain = qty - close_qty
            no_cost = remain * (1 - price)
            if self.no_qty > 0:
                total = self.no_avg_cost * self.no_qty + no_cost
                self.no_qty += remain
                self.no_avg_cost = total / self.no_qty
            else:
                self.no_qty = remain
                self.no_avg_cost = 1 - price
            return close_qty * price
        else:
            # Open NO position
            no_cost = qty * (1 - price)
            if self.no_qty > 0:
                total = self.no_avg_cost * self.no_qty + no_cost
                self.no_qty += qty
                self.no_avg_cost = total / self.no_qty
            else:
                self.no_qty = qty
                self.no_avg_cost = 1 - price
            return 0  # no immediate revenue

    def unrealized_pnl(self, mid: float) -> float:
        """Calculate unrealized P&L at current mid price."""
        pnl = 0.0
        if self.yes_qty > 0:
            pnl += self.yes_qty * (mid - self.yes_avg_cost)
        if self.no_qty > 0:
            pnl += self.no_qty * ((1 - mid) - self.no_avg_cost)
        return pnl

    def liquidate(self, mid: float) -> float:
        """Close all positions at mid price."""
        pnl = self.unrealized_pnl(mid)
        self.realized_pnl += pnl
        self.yes_qty = 0
        self.no_qty = 0
        return pnl

    @property
    def total_exposure(self) -> float:
        return self.yes_qty + self.no_qty

    def capital_locked(self, mid: float) -> float:
        """Capital currently locked in positions."""
        yes_locked = self.yes_qty * self.yes_avg_cost if self.yes_qty > 0 else 0
        no_locked = self.no_qty * self.no_avg_cost if self.no_qty > 0 else 0
        return yes_locked + no_locked


@dataclass
class MarketState:
    """State for a single market being traded."""
    condition_id: str
    token_id: str
    question: str
    daily_rate: float
    min_size: int
    max_spread: float

    position: Position = field(default_factory=Position)

    # Order state
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    bid_live: bool = False
    ask_live: bool = False
    bid_cooldown_until: float = 0.0
    ask_cooldown_until: float = 0.0

    # Stats
    bid_fills: int = 0
    ask_fills: int = 0
    bid_on_book_sec: float = 0.0
    ask_on_book_sec: float = 0.0
    reprices: int = 0
    gas_cost: float = 0.0

    # For queue tracking
    queue_position_bid: float = 0.0
    queue_position_ask: float = 0.0

    last_mid: float = 0.5


@dataclass
class Wallet:
    """Track available capital."""
    initial_capital: float
    available: float = 0.0

    def __post_init__(self):
        self.available = self.initial_capital

    def can_afford(self, amount: float) -> bool:
        return self.available >= amount

    def lock(self, amount: float) -> bool:
        if self.can_afford(amount):
            self.available -= amount
            return True
        return False

    def unlock(self, amount: float):
        self.available += amount

    @property
    def locked(self) -> float:
        return self.initial_capital - self.available


def fetch_sampling_markets(http: httpx.Client) -> list:
    """Fetch markets with rewards."""
    try:
        resp = http.get(f"{CLOB_BASE}/sampling-markets", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data or []
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def fetch_trades(http: httpx.Client, condition_id: str, limit: int = 500) -> list:
    """Fetch historical trades."""
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
        return []


def select_affordable_markets(http: httpx.Client, capital: float, top_n: int = 5) -> list:
    """Select markets that can be traded with given capital."""
    print(f"Fetching markets (capital: ${capital})...")
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

            if daily_rate < 0.001 or max_spread < 1:
                continue

            tokens = m.get("tokens") or []
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            if not yes_token:
                continue

            price = float(yes_token.get("price", 0.5))
            if not (0.05 <= price <= 0.95):
                continue

            # Calculate capital needed for this market
            # To place both bid and ask, need:
            # - Bid side: min_size * bid_price (to buy YES)
            # - Ask side: min_size * (1 - ask_price) (to buy NO as collateral)
            bid_price = price - (max_spread * 0.15 / 100)  # 15% of max spread
            ask_price = price + (max_spread * 0.15 / 100)
            bid_capital = min_size * bid_price
            ask_capital = min_size * (1 - ask_price)
            total_capital_needed = bid_capital + ask_capital

            # Skip if we can't afford even one side
            min_capital = min(bid_capital, ask_capital)
            if min_capital > capital:
                continue

            candidates.append({
                "condition_id": m.get("condition_id"),
                "question": (m.get("question") or "")[:55],
                "token_id": yes_token.get("token_id"),
                "daily_rate": daily_rate,
                "min_size": min_size,
                "max_spread": max_spread,
                "price": price,
                "capital_needed": total_capital_needed,
                "bid_capital": bid_capital,
                "ask_capital": ask_capital,
                "can_both_sides": total_capital_needed <= capital,
            })
        except Exception:
            continue

    # Sort by daily_rate / capital_needed (efficiency)
    for c in candidates:
        c["efficiency"] = c["daily_rate"] / max(c["capital_needed"], 1)
    candidates.sort(key=lambda x: -x["efficiency"])

    affordable = [c for c in candidates if c["can_both_sides"]]
    one_side_only = [c for c in candidates if not c["can_both_sides"]]

    print(f"  {len(affordable)} markets affordable (both sides)")
    print(f"  {len(one_side_only)} markets one-side only")

    # Prefer both-side markets, then add one-side if needed
    selected = affordable[:top_n]
    if len(selected) < top_n:
        selected.extend(one_side_only[:top_n - len(selected)])

    return selected


def simulate_market(trades: list, market: dict, wallet: Wallet,
                    spread_pct: float = 0.15) -> dict:
    """Simulate trading with capital constraints."""
    if len(trades) < 10:
        return None

    min_size = market["min_size"]
    max_spread = market["max_spread"]
    daily_rate = market["daily_rate"]

    t_start = trades[0]["timestamp"]
    t_end = trades[-1]["timestamp"]
    duration_sec = t_end - t_start
    n_days = max(duration_sec / 86400, 0.001)

    # Estimate queue depth from trade sizes
    trade_sizes = [t["size"] for t in trades if t["size"] > 0]
    avg_trade_size = statistics.mean(trade_sizes) if trade_sizes else min_size
    queue_depth = avg_trade_size * QUEUE_DEPTH_MULTIPLIER

    state = MarketState(
        condition_id=market["condition_id"],
        token_id=market["token_id"],
        question=market["question"],
        daily_rate=daily_rate,
        min_size=min_size,
        max_spread=max_spread,
    )

    ema_mid = trades[0]["price"]
    last_reprice_mid = ema_mid
    last_ts = t_start

    # Capital tracking
    bid_capital_locked = 0.0
    ask_capital_locked = 0.0

    for trade in trades:
        ts = trade["timestamp"]
        price = trade["price"]
        size = trade["size"] or min_size
        side = trade["side"]

        if price <= 0.01 or price >= 0.99:
            continue

        dt = ts - last_ts

        # Accumulate on-book time
        if state.bid_live:
            state.bid_on_book_sec += dt
        if state.ask_live:
            state.ask_on_book_sec += dt

        # Cooldown check
        if not state.bid_live and ts >= state.bid_cooldown_until:
            # Can we afford to place bid?
            bid_cost = min_size * (ema_mid - max_spread * spread_pct / 100)
            bid_cost = max(bid_cost, min_size * 0.01)
            if wallet.lock(bid_cost):
                state.bid_live = True
                state.bid_size = min_size
                state.queue_position_bid = queue_depth  # join at back
                bid_capital_locked = bid_cost

        if not state.ask_live and ts >= state.ask_cooldown_until:
            ask_price = ema_mid + max_spread * spread_pct / 100
            ask_cost = min_size * (1 - ask_price)  # NO collateral
            ask_cost = max(ask_cost, min_size * 0.01)
            if wallet.lock(ask_cost):
                state.ask_live = True
                state.ask_size = min_size
                state.queue_position_ask = queue_depth
                ask_capital_locked = ask_cost

        # Calculate our order prices
        dist_c = max_spread * spread_pct
        half = max(dist_c / 100, 0.001)
        our_bid = max(0.01, ema_mid - half)
        our_ask = min(0.99, ema_mid + half)
        state.bid_price = our_bid
        state.ask_price = our_ask

        # Reprice check
        if abs(ema_mid - last_reprice_mid) >= 0.005:
            last_reprice_mid = ema_mid
            state.reprices += 1
            state.gas_cost += 4 * GAS_PER_TX  # cancel + place × 2 sides
            # Reset queue position on reprice
            state.queue_position_bid = queue_depth
            state.queue_position_ask = queue_depth

        # === FILL CHECK ===
        # Model: trades at our level consume queue ahead of us

        if side == "SELL" and price <= our_bid + 0.003 and state.bid_live:
            # Sell trade near our bid - queue advances
            state.queue_position_bid = max(0, state.queue_position_bid - size)

            # If queue consumed, we might get filled
            if state.queue_position_bid <= 0:
                fill_prob = min_size / (min_size + queue_depth * 0.3)  # residual queue
                if random.random() < fill_prob:
                    # FILLED on bid - we bought YES
                    cost = state.position.buy_yes(min_size, our_bid)
                    wallet.unlock(bid_capital_locked)  # release locked capital
                    bid_capital_locked = 0

                    state.bid_fills += 1
                    state.bid_live = False
                    state.bid_cooldown_until = ts + COOLDOWN_SEC

                    # Queue reset for next order
                    state.queue_position_bid = queue_depth

        if side == "BUY" and price >= our_ask - 0.003 and state.ask_live:
            state.queue_position_ask = max(0, state.queue_position_ask - size)

            if state.queue_position_ask <= 0:
                fill_prob = min_size / (min_size + queue_depth * 0.3)
                if random.random() < fill_prob:
                    # FILLED on ask - we sold YES (or bought NO)
                    revenue = state.position.sell_yes(min_size, our_ask)
                    wallet.unlock(ask_capital_locked)
                    if revenue > 0:
                        wallet.unlock(revenue)  # add revenue back
                    ask_capital_locked = 0

                    state.ask_fills += 1
                    state.ask_live = False
                    state.ask_cooldown_until = ts + COOLDOWN_SEC

                    state.queue_position_ask = queue_depth

        # Update EMA mid
        ema_mid = 0.3 * price + 0.7 * ema_mid
        state.last_mid = ema_mid
        last_ts = ts

    # End simulation - liquidate any position
    liq_pnl = state.position.liquidate(ema_mid)
    wallet.unlock(bid_capital_locked + ask_capital_locked)

    # Calculate reward
    avg_on_book = min(state.bid_on_book_sec, state.ask_on_book_sec)
    on_book_ratio = avg_on_book / max(duration_sec, 1)

    # Q score
    dist_c = max_spread * spread_pct
    our_q = ((max_spread - dist_c) / max_spread) ** 2 * min_size if dist_c < max_spread else 0
    total_q = 100  # estimate
    our_share = our_q / (total_q + our_q)
    reward = daily_rate * n_days * our_share * on_book_ratio

    net = reward + state.position.realized_pnl - state.gas_cost

    return {
        "question": market["question"],
        "bid_fills": state.bid_fills,
        "ask_fills": state.ask_fills,
        "total_fills": state.bid_fills + state.ask_fills,
        "reward": reward,
        "realized_pnl": state.position.realized_pnl,
        "gas": state.gas_cost,
        "net": net,
        "n_days": n_days,
        "n_trades": len(trades),
        "on_book_ratio": on_book_ratio,
        "bid_on_book_pct": state.bid_on_book_sec / max(duration_sec, 1) * 100,
        "ask_on_book_pct": state.ask_on_book_sec / max(duration_sec, 1) * 100,
        "capital_needed": market["capital_needed"],
        "both_sides": market["can_both_sides"],
    }


def run_backtest(capital: float = 100, top_n: int = 5):
    """Run realistic backtest with fixed capital."""
    print(f"\n{'='*60}")
    print(f"REALISTIC BACKTEST - ${capital} Capital")
    print(f"{'='*60}\n")

    http = httpx.Client(timeout=30)
    wallet = Wallet(initial_capital=capital)

    try:
        markets = select_affordable_markets(http, capital, top_n * 2)
        if not markets:
            print("No affordable markets found!")
            return

        print(f"\nTesting {min(len(markets), top_n)} markets:\n")

        results = []
        for i, m in enumerate(markets[:top_n]):
            print(f"[{i+1}] {m['question']}")
            print(f"    Rate: ${m['daily_rate']:.2f}/d | min_size: {m['min_size']} | "
                  f"cap needed: ${m['capital_needed']:.0f} | both sides: {m['can_both_sides']}")

            trades = fetch_trades(http, m["condition_id"])
            if len(trades) < 10:
                print(f"    Skipped: only {len(trades)} trades")
                continue

            # Reset wallet for each market (simulate running one at a time)
            wallet = Wallet(initial_capital=capital)

            res = simulate_market(trades, m, wallet, spread_pct=0.15)
            if res:
                results.append(res)
                print(f"    {res['n_trades']} trades over {res['n_days']:.1f} days")
                print(f"    Fills: bid={res['bid_fills']}, ask={res['ask_fills']}")
                print(f"    On-book: bid={res['bid_on_book_pct']:.0f}%, ask={res['ask_on_book_pct']:.0f}%")
                print(f"    NET: ${res['net']:.2f} (reward: ${res['reward']:.2f}, "
                      f"fill P&L: ${res['realized_pnl']:.2f})")

            time.sleep(0.2)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        if not results:
            print("No markets successfully backtested.")
            return

        total_net = sum(r["net"] for r in results)
        total_reward = sum(r["reward"] for r in results)
        total_bid_fills = sum(r["bid_fills"] for r in results)
        total_ask_fills = sum(r["ask_fills"] for r in results)
        total_gas = sum(r["gas"] for r in results)
        avg_days = sum(r["n_days"] for r in results) / len(results)

        both_sides_markets = sum(1 for r in results if r["both_sides"])
        one_side_markets = len(results) - both_sides_markets

        print(f"Capital:            ${capital}")
        print(f"Markets tested:     {len(results)} ({both_sides_markets} both-sides, {one_side_markets} one-side)")
        print(f"Average period:     {avg_days:.1f} days")
        print()
        print(f"Total NET:          ${total_net:.2f}")
        print(f"  Reward:           ${total_reward:.2f}")
        print(f"  Fill P&L:         ${sum(r['realized_pnl'] for r in results):.2f}")
        print(f"  Gas:              ${total_gas:.2f}")
        print()
        print(f"Fills:              {total_bid_fills + total_ask_fills} (bid: {total_bid_fills}, ask: {total_ask_fills})")

        # Imbalance analysis
        if total_bid_fills + total_ask_fills > 0:
            imbalance = abs(total_bid_fills - total_ask_fills) / (total_bid_fills + total_ask_fills)
            print(f"Fill imbalance:     {imbalance:.1%} {'(PROBLEM!)' if imbalance > 0.3 else ''}")

        # ROI
        if avg_days > 0:
            daily_roi = (total_net / capital) / avg_days * 100
            print()
            print(f"Daily ROI:          {daily_roi:.2f}%")
            print(f"Monthly ROI:        {daily_roi * 30:.1f}%")

        # Per-market
        print(f"\n--- Per Market ---")
        for r in sorted(results, key=lambda x: -x["net"]):
            imb = "balanced" if abs(r["bid_fills"] - r["ask_fills"]) <= 2 else "IMBALANCED"
            print(f"  {r['question'][:45]}")
            print(f"    NET: ${r['net']:>6.2f} | fills: {r['total_fills']:>2} (b:{r['bid_fills']}/a:{r['ask_fills']}) | {imb}")

    finally:
        http.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realistic Polymarket Backtest")
    parser.add_argument("--capital", type=float, default=100, help="Starting capital in USDC")
    parser.add_argument("--top", type=int, default=5, help="Number of markets to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    run_backtest(capital=args.capital, top_n=args.top)
