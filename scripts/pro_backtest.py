"""Professional backtest based on defiance_cr + poly-maker + official MM strategies.

Key learnings applied:
- Order distance = 15% of max_spread (defiance_cr's formula)
- Market selection: gm_reward_per_100, volatility filter
- Risk: stop-loss, vol gate, book imbalance, position cap
- Position sizing: trade_size scales with reward, max_size = 2x trade
- Fills trigger cooldown + re-hedge

Uses REAL trade data from data-api.polymarket.com for fill simulation.
"""
import sys
sys.path.insert(0, "/workspace/mantis")

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def q_score(size, dist_c, max_spread_c):
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


@dataclass
class StrategyParams:
    name: str
    # Order placement
    spread_pct: float       # distance = spread_pct * max_spread (defiance=0.15)
    # Position sizing
    trade_size_base: int    # base order size
    max_size_mult: float    # max_size = trade_size * mult
    # Risk
    stop_loss_pct: float    # -2% default
    vol_gate: bool          # pause on high vol
    book_imbalance: bool    # check bid/ask ratio
    cooldown_sec: int       # seconds after fill before re-placing
    # Reward gate
    min_reward_pct: float   # min gm_reward_per_100 to trade


STRATEGIES = {
    "naive": StrategyParams(
        name="Naive 0.1c", spread_pct=0.0, trade_size_base=50,
        max_size_mult=1.0, stop_loss_pct=-99, vol_gate=False,
        book_imbalance=False, cooldown_sec=30, min_reward_pct=0,
    ),
    "defiance": StrategyParams(
        name="Defiance (15% max_spread)", spread_pct=0.15, trade_size_base=50,
        max_size_mult=2.0, stop_loss_pct=-2.0, vol_gate=True,
        book_imbalance=True, cooldown_sec=60, min_reward_pct=0.5,
    ),
    "conservative": StrategyParams(
        name="Conservative (25%)", spread_pct=0.25, trade_size_base=30,
        max_size_mult=2.0, stop_loss_pct=-3.0, vol_gate=True,
        book_imbalance=True, cooldown_sec=120, min_reward_pct=1.0,
    ),
}


def calc_order_distance(strategy, max_spread_c, mid):
    """Calculate order distance from mid in price units."""
    if strategy.spread_pct == 0:
        return 0.001  # naive: 0.1c fixed
    return (strategy.spread_pct * max_spread_c) / 100


def run_backtest_on_market(conn, cid, question, daily_rate, max_spread,
                           capital, strategy: StrategyParams):
    """Run backtest for one market with one strategy using real trades."""
    trades = conn.execute(
        "SELECT side, price, size, timestamp FROM trades "
        "WHERE condition_id=? ORDER BY timestamp ASC", (cid,)
    ).fetchall()

    snaps = conn.execute(
        "SELECT ts_unix, best_bid, best_ask, midpoint, spread_cents, total_q "
        "FROM snapshots WHERE condition_id=? ORDER BY ts_unix ASC", (cid,)
    ).fetchall()

    if len(trades) < 5 or not snaps:
        return None

    t_start, t_end = trades[0][3], trades[-1][3]
    n_days = max((t_end - t_start) / 86400, 0.1)

    avg_mid = statistics.mean(s[3] for s in snaps if s[3] and s[3] > 0)
    avg_q = statistics.mean(s[5] for s in snaps if s[5] is not None)
    max_spread_c = max_spread

    # Position sizing based on capital
    trade_size = min(strategy.trade_size_base, int(capital / avg_mid / 2))
    trade_size = max(trade_size, 10)  # floor
    max_size = int(trade_size * strategy.max_size_mult)

    # Order distance
    half = calc_order_distance(strategy, max_spread_c, avg_mid)
    half_c = half * 100

    # Reward check: would we even trade this market?
    our_q = q_score(trade_size, half_c, max_spread_c)
    total_q = max(avg_q, 1.0) + our_q
    our_share = our_q / total_q
    est_daily_reward = daily_rate * our_share
    reward_per_100 = est_daily_reward / (trade_size * avg_mid * 2) * 100

    if strategy.min_reward_pct > 0 and reward_per_100 < strategy.min_reward_pct:
        return None  # skip market

    # Simulate
    fills = 0
    fill_cost = 0.0
    position = 0  # net YES position
    avg_price = 0.0
    bid_live = True
    ask_live = True
    bid_cooldown_until = 0
    ask_cooldown_until = 0
    stopped_out = False
    stop_loss_cost = 0.0

    snap_idx = 0
    recent_prices = []

    for side, price, size, ts in trades:
        if stopped_out:
            break

        # Cooldowns
        if not bid_live and ts >= bid_cooldown_until:
            bid_live = True
        if not ask_live and ts >= ask_cooldown_until:
            ask_live = True

        # Update snap
        while snap_idx < len(snaps) - 1 and snaps[snap_idx + 1][0] <= ts:
            snap_idx += 1
        snap = snaps[min(snap_idx, len(snaps) - 1)]
        snap_mid = snap[3] if snap[3] and snap[3] > 0 else avg_mid

        # Vol gate: track recent prices
        if strategy.vol_gate:
            recent_prices.append(price)
            if len(recent_prices) > 50:
                recent_prices = recent_prices[-50:]
            if len(recent_prices) >= 10:
                returns = [abs(recent_prices[i] - recent_prices[i-1])
                          for i in range(1, len(recent_prices))]
                recent_vol = statistics.mean(returns) * 100
                if recent_vol > 2.0:  # high vol = pause
                    bid_live = False
                    bid_cooldown_until = ts + 300
                    continue

        our_bid = snap_mid - half
        our_ask = snap_mid + half

        # Position cap
        if position >= max_size:
            bid_live = False

        # Fill check
        if side == "SELL" and price <= our_bid and price > 0 and bid_live:
            adverse = half * trade_size
            fill_cost += adverse
            fills += 1
            # Update position
            if position == 0:
                avg_price = our_bid
            else:
                avg_price = (avg_price * position + our_bid * trade_size) / (position + trade_size)
            position += trade_size
            bid_live = False
            bid_cooldown_until = ts + strategy.cooldown_sec

            # Stop-loss check
            if avg_price > 0:
                pnl_pct = (snap_mid - avg_price) / avg_price * 100
                if pnl_pct < strategy.stop_loss_pct:
                    # Sell everything at best bid
                    stop_loss_cost += max(0, (avg_price - snap_mid + 0.01)) * position
                    position = 0
                    avg_price = 0
                    stopped_out = True

        elif side == "BUY" and price >= our_ask and price < 1 and ask_live:
            adverse = half * trade_size
            fill_cost += adverse
            fills += 1
            ask_live = False
            ask_cooldown_until = ts + strategy.cooldown_sec

    # Reward calculation
    # Uptime penalty: fills cause downtime
    total_ticks = len(trades)
    downtime_ticks = fills * (strategy.cooldown_sec / max((t_end - t_start) / total_ticks, 1))
    uptime = max(0, 1 - downtime_ticks / total_ticks)

    eff_q = our_q * uptime
    total_q_eff = max(avg_q, 1.0) + eff_q
    share = eff_q / total_q_eff
    total_reward = daily_rate * n_days * share

    gas = fills * 0.005
    net = total_reward - fill_cost - gas - stop_loss_cost

    return {
        "question": question[:40],
        "daily_rate": daily_rate,
        "n_days": n_days,
        "n_trades": len(trades),
        "avg_q": avg_q,
        "half_c": half_c,
        "trade_size": trade_size,
        "fills": fills,
        "fill_cost": fill_cost,
        "stop_loss_cost": stop_loss_cost,
        "reward": total_reward,
        "gas": gas,
        "net": net,
        "reward_per_100": reward_per_100,
        "uptime": uptime,
        "net_per_day": net / n_days if n_days > 0 else 0,
    }


def run_all(capital: float):
    """Run all strategies across all markets for given capital."""
    conn = sqlite3.connect("/workspace/mantis/data/snapshots.db")
    markets = conn.execute("""
        SELECT m.question, m.condition_id, m.daily_rate, m.max_spread
        FROM markets m ORDER BY m.daily_rate DESC
    """).fetchall()

    results = {}
    for sname, strategy in STRATEGIES.items():
        results[sname] = []
        for question, cid, daily_rate, max_spread in markets:
            r = run_backtest_on_market(conn, cid, question, daily_rate, max_spread,
                                       capital, strategy)
            if r:
                results[sname].append(r)

    conn.close()
    return results


def main():
    capitals = [100, 500, 1000, 5000, 10000]

    print("=" * 80)
    print("PROFESSIONAL BACKTEST: defiance_cr strategy vs Naive vs Conservative")
    print("Based on real Polymarket trade data + orderbook snapshots")
    print("=" * 80)

    all_results = {}
    for cap in capitals:
        print(f"\n--- Capital: ${cap} ---")
        results = run_all(cap)
        all_results[cap] = results

        for sname, sresults in results.items():
            if not sresults:
                print(f"  {STRATEGIES[sname].name}: NO MARKETS (filtered out)")
                continue
            total_net = sum(r["net"] for r in sresults)
            total_fills = sum(r["fills"] for r in sresults)
            total_reward = sum(r["reward"] for r in sresults)
            n_markets = len(sresults)
            profitable = sum(1 for r in sresults if r["net"] > 0)
            print(f"  {STRATEGIES[sname].name}: {n_markets} mkts | "
                  f"NET ${total_net:.1f} | reward ${total_reward:.1f} | "
                  f"fills {total_fills} | profitable {profitable}/{n_markets}")

    # ── Generate chart ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes.flat:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#f0f6fc")

    colors = {"naive": "#ff6b35", "defiance": "#00d4ff", "conservative": "#3fb950"}

    # Panel 1: NET by capital level
    ax = axes[0, 0]
    x = np.arange(len(capitals))
    w = 0.25
    for i, (sname, strat) in enumerate(STRATEGIES.items()):
        nets = []
        for cap in capitals:
            sresults = all_results[cap].get(sname, [])
            nets.append(sum(r["net"] for r in sresults) if sresults else 0)
        ax.bar(x + i * w, nets, w, color=colors[sname], alpha=0.8, label=strat.name)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"${c}" for c in capitals], color="#c9d1d9")
    ax.set_ylabel("Total NET $")
    ax.set_title("NET Profit by Capital Level", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax.axhline(0, color="#f85149", lw=1, ls="--", alpha=0.5)

    # Panel 2: NET per day by capital
    ax = axes[0, 1]
    for sname, strat in STRATEGIES.items():
        daily_nets = []
        for cap in capitals:
            sresults = all_results[cap].get(sname, [])
            if sresults:
                total_days = max(r["n_days"] for r in sresults)
                total_net = sum(r["net"] for r in sresults)
                daily_nets.append(total_net / total_days)
            else:
                daily_nets.append(0)
        ax.plot(capitals, daily_nets, 'o-', color=colors[sname], lw=2, ms=8, label=strat.name)
    ax.set_xlabel("Capital $")
    ax.set_ylabel("NET $/day")
    ax.set_title("Daily Return by Capital", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax.axhline(0, color="#f85149", lw=1, ls="--", alpha=0.5)

    # Panel 3: Fills comparison
    ax = axes[1, 0]
    for i, (sname, strat) in enumerate(STRATEGIES.items()):
        fills = []
        for cap in capitals:
            sresults = all_results[cap].get(sname, [])
            fills.append(sum(r["fills"] for r in sresults) if sresults else 0)
        ax.bar(x + i * w, fills, w, color=colors[sname], alpha=0.8, label=strat.name)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"${c}" for c in capitals], color="#c9d1d9")
    ax.set_ylabel("Total Fills")
    ax.set_title("Fills by Capital (fewer = less adverse selection)", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis("off")

    lines = [
        ("STRATEGY COMPARISON (from open-source research)", "#f0f6fc", 12, "bold"),
        ("", "", 0, ""),
    ]

    # Show $1000 capital as reference
    ref_cap = 1000
    ref = all_results.get(ref_cap, {})
    lines.append((f"Reference: ${ref_cap} capital", "#8b949e", 10, "bold"))
    lines.append(("-" * 55, "#30363d", 9, "normal"))

    for sname, strat in STRATEGIES.items():
        sresults = ref.get(sname, [])
        if sresults:
            tn = sum(r["net"] for r in sresults)
            tr = sum(r["reward"] for r in sresults)
            tf = sum(r["fills"] for r in sresults)
            nm = len(sresults)
            td = max(r["n_days"] for r in sresults)
            lines.append((f"{strat.name}", colors[sname], 11, "bold"))
            lines.append((f"  Markets: {nm} | NET ${tn:.1f} | ${tn/td:.2f}/day", "#c9d1d9", 9, "normal"))
            lines.append((f"  Reward: ${tr:.1f} | Fills: {tf} | Size: {sresults[0]['trade_size']}", "#8b949e", 9, "normal"))
        else:
            lines.append((f"{strat.name}: SKIPPED (reward too low)", "#f85149", 10, "normal"))
        lines.append(("", "", 0, ""))

    lines.append(("-" * 55, "#30363d", 9, "normal"))
    lines.append(("DEFIANCE_CR KEY PARAMS:", "#00d4ff", 11, "bold"))
    lines.append(("  Distance = 15% of max_spread", "#8b949e", 9, "normal"))
    lines.append(("  Q coefficient = 72% (vs 95% at 0.1c)", "#8b949e", 9, "normal"))
    lines.append(("  Stop-loss: -2% PnL", "#8b949e", 9, "normal"))
    lines.append(("  Vol gate: pause if 3h vol > threshold", "#8b949e", 9, "normal"))
    lines.append(("  Reward filter: gm_reward >= 0.5%/100$", "#8b949e", 9, "normal"))
    lines.append(("  Cooldown: 60s after fill", "#8b949e", 9, "normal"))
    lines.append(("  Max position: 2x trade_size, cap 250", "#8b949e", 9, "normal"))

    y = 0.98
    for text, color, size, weight in lines:
        if size > 0:
            ax.text(0.02, y, text, transform=ax.transAxes, fontsize=size,
                    fontweight=weight, color=color, fontfamily="monospace")
        y -= 0.04

    fig.suptitle("Mantis Pro Backtest: Real Data x Open-Source Strategy Research\n"
                 "defiance_cr + poly-maker + official Polymarket MM insights",
                 color="#f0f6fc", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = "/workspace/mantis/reports/pro_backtest_multi_capital.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nChart: {out}")


if __name__ == "__main__":
    main()
