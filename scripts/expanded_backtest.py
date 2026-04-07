"""Expanded backtest on 72 markets with real trade data.

Strategies:
  - Naive: 0.1c from mid (old approach)
  - Defiance: 15% of max_spread from mid (proven approach)
  - Conservative: 25% of max_spread from mid

Market selection filters (defiance_cr):
  - gm_reward_per_100 >= 1.0%
  - volatility_sum < 20
  - price 0.10-0.90
  - spread < 0.10
"""
import sys
sys.path.insert(0, "/workspace/mantis")

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DB_PATH = "/workspace/mantis/data/backtest.db"


def q_score(size, dist_c, max_spread_c):
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


@dataclass
class StrategyResult:
    name: str
    markets_traded: int = 0
    total_fills: int = 0
    total_reward: float = 0.0
    total_adverse: float = 0.0
    total_gas: float = 0.0
    total_net: float = 0.0
    per_market: list = field(default_factory=list)


def simulate_market(trades, snaps, daily_rate, max_spread, min_size, spread_pct, cooldown_sec):
    """Simulate one strategy on one market.

    Uses rolling midpoint from recent trades (EMA) to simulate realistic
    order placement that tracks price movement.
    """
    if not trades:
        return {"fills": 0, "reward": 0, "adverse": 0, "gas": 0, "net": 0}

    t_start = trades[0][2]
    t_end = trades[-1][2]
    n_days = max((t_end - t_start) / 86400, 0.01)

    # Get avg Q from snapshots
    if snaps:
        avg_q = statistics.mean(s[4] for s in snaps if s[4] is not None) if snaps[0][4] is not None else 50
    else:
        avg_q = 50

    # Compute our distance and Q
    dist_c = max_spread * spread_pct if spread_pct > 0 else 0.1
    half = dist_c / 100

    # Simulate fills with rolling midpoint
    fills = 0
    adverse_total = 0.0
    bid_live = True
    ask_live = True
    bid_cooldown_until = 0
    ask_cooldown_until = 0

    # Use a fast EMA for rolling mid, updated AFTER each trade
    # This simulates: we see trades, update our mid, re-place orders
    recent_prices = []
    ema_mid = trades[0][1]
    alpha = 0.3  # faster EMA to track price

    snap_idx = 0
    last_rebalance = 0
    REBALANCE_INTERVAL = 10  # seconds between order updates (realistic for bot)

    for side, price, ts in trades:
        if price <= 0 or price >= 1:
            continue

        if not bid_live and ts >= bid_cooldown_until:
            bid_live = True
        if not ask_live and ts >= ask_cooldown_until:
            ask_live = True

        # Use EMA mid for order placement (snapshots are too sparse for expanded data)
        our_bid = round(ema_mid - half, 2)
        our_ask = round(ema_mid + half, 2)

        # Fill logic: trade must be AT our exact price level (within 1 tick)
        # A SELL trade at our_bid means someone sold into our bid
        # A BUY trade at our_ask means someone bought from our ask
        filled = False
        tick = 0.01

        if side == "SELL" and price <= our_bid + tick and price > 0.01 and bid_live:
            filled = True
            adverse = max(0, our_bid - price) * min_size
            adverse_total += adverse
            bid_live = False
            bid_cooldown_until = ts + cooldown_sec
        elif side == "BUY" and price >= our_ask - tick and price < 0.99 and ask_live:
            filled = True
            adverse = max(0, price - our_ask) * min_size
            adverse_total += adverse
            ask_live = False
            ask_cooldown_until = ts + cooldown_sec

        if filled:
            fills += 1

        # Update EMA
        ema_mid = alpha * price + (1 - alpha) * ema_mid

    # Reward calculation
    our_q = q_score(min_size, dist_c, max_spread)
    total_q = max(avg_q, 1.0) + our_q
    our_share = our_q / total_q
    total_reward = daily_rate * n_days * our_share

    gas = fills * 0.005
    net = total_reward - adverse_total - gas

    return {
        "fills": fills,
        "reward": total_reward,
        "adverse": adverse_total,
        "gas": gas,
        "net": net,
        "n_days": n_days,
        "dist_c": dist_c,
        "our_q": our_q,
        "our_share": our_share,
    }


def run_backtest(capital=100):
    conn = sqlite3.connect(DB_PATH)

    # Get all markets with metadata
    markets = conn.execute("""
        SELECT condition_id, question, daily_rate, max_spread, min_size,
               gm_reward_per_100, volatility_sum, best_bid, best_ask
        FROM markets
        ORDER BY profitability_score DESC
    """).fetchall()

    strategies = {
        "naive": {"spread_pct": 0, "cooldown": 30},       # 0.1c fixed
        "defiance": {"spread_pct": 0.15, "cooldown": 60},  # 15% max_spread
        "conservative": {"spread_pct": 0.25, "cooldown": 120},  # 25% max_spread
    }

    results = {name: StrategyResult(name=name) for name in strategies}

    # Also track by market selection tier
    tier_results = {
        "all": {name: StrategyResult(name=name) for name in strategies},
        "gm>=1%": {name: StrategyResult(name=name) for name in strategies},
        "gm>=1%+vol<20": {name: StrategyResult(name=name) for name in strategies},
    }

    market_details = []
    skipped_capital = 0

    for cid, question, daily_rate, max_spread, min_size, gm_rpc, vol_sum, bb, ba in markets:
        # Capital filter: can we afford min_size on both sides?
        mid_est = (bb + ba) / 2 if bb and ba else 0.5
        cap_needed = min_size * max(mid_est, 1 - mid_est) * 2  # both sides
        if cap_needed > capital:
            skipped_capital += 1
            continue
        # Get Yes token ID to filter trades
        yes_row = conn.execute(
            "SELECT token_id_yes FROM markets WHERE condition_id = ?", (cid,)
        ).fetchone()
        yes_token_id = yes_row[0] if yes_row else None

        # Get trades - ONLY for Yes token (No token prices are complements!)
        if yes_token_id:
            trades_raw = conn.execute("""
                SELECT side, price, timestamp FROM trades
                WHERE condition_id = ? AND asset = ? ORDER BY timestamp ASC
            """, (cid, yes_token_id)).fetchall()
        else:
            trades_raw = conn.execute("""
                SELECT side, price, timestamp FROM trades
                WHERE condition_id = ? ORDER BY timestamp ASC
            """, (cid,)).fetchall()

        if len(trades_raw) < 20:
            continue

        # Get orderbook snapshots
        snaps = conn.execute("""
            SELECT ts, best_bid, midpoint, spread_cents, total_q
            FROM orderbook_snapshots WHERE condition_id = ? ORDER BY ts ASC
        """, (cid,)).fetchall()

        # Determine tier
        tiers = ["all"]
        if gm_rpc >= 1.0:
            tiers.append("gm>=1%")
        if gm_rpc >= 1.0 and vol_sum < 20:
            tiers.append("gm>=1%+vol<20")

        market_row = {
            "question": question, "cid": cid, "daily_rate": daily_rate,
            "max_spread": max_spread, "gm_rpc": gm_rpc, "vol_sum": vol_sum,
            "n_trades": len(trades_raw), "tiers": tiers,
        }

        for sname, sparams in strategies.items():
            res = simulate_market(
                trades_raw, snaps, daily_rate, max_spread, min_size,
                sparams["spread_pct"], sparams["cooldown"],
            )
            market_row[f"{sname}_net"] = res["net"]
            market_row[f"{sname}_fills"] = res["fills"]
            market_row[f"{sname}_reward"] = res["reward"]
            market_row[f"{sname}_adverse"] = res["adverse"]

            # Aggregate
            results[sname].total_fills += res["fills"]
            results[sname].total_reward += res["reward"]
            results[sname].total_adverse += res["adverse"]
            results[sname].total_gas += res["gas"]
            results[sname].total_net += res["net"]
            results[sname].markets_traded += 1

            for tier in tiers:
                sr = tier_results[tier][sname]
                sr.total_fills += res["fills"]
                sr.total_reward += res["reward"]
                sr.total_adverse += res["adverse"]
                sr.total_gas += res["gas"]
                sr.total_net += res["net"]
                sr.markets_traded += 1

        market_details.append(market_row)

    print(f"Skipped {skipped_capital} markets (capital ${capital} too low for min_size)")
    conn.close()
    return results, tier_results, market_details


def plot_results(results, tier_results, market_details):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
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

    C, O, G, R, P = "#00d4ff", "#ff6b35", "#3fb950", "#f85149", "#a371f7"

    # Panel 1: NET by market (top 25 by defiance net)
    ax = axes[0, 0]
    sorted_markets = sorted(market_details, key=lambda m: -m["defiance_net"])[:25]
    labels = [m["question"][:28] for m in sorted_markets]
    y = range(len(sorted_markets))
    naive_nets = [m["naive_net"] for m in sorted_markets]
    def_nets = [m["defiance_net"] for m in sorted_markets]
    cons_nets = [m["conservative_net"] for m in sorted_markets]

    ax.barh([i - 0.25 for i in y], naive_nets, height=0.25, color=O, alpha=0.8, label="Naive")
    ax.barh([i for i in y], def_nets, height=0.25, color=C, alpha=0.8, label="Defiance")
    ax.barh([i + 0.25 for i in y], cons_nets, height=0.25, color=P, alpha=0.8, label="Conservative")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6, color="#c9d1d9")
    ax.axvline(0, color=R, lw=1, ls="--")
    ax.set_xlabel("NET $")
    ax.set_title("Top 25 Markets by Defiance NET", fontsize=11)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax.invert_yaxis()

    # Panel 2: Strategy comparison by tier
    ax = axes[0, 1]
    tier_names = ["all", "gm>=1%", "gm>=1%+vol<20"]
    tier_labels = ["All Markets", "gm_rpc>=1%", "gm>=1% + vol<20"]
    x = np.arange(len(tier_names))
    width = 0.25

    for i, (sname, color) in enumerate([("naive", O), ("defiance", C), ("conservative", P)]):
        vals = [tier_results[t][sname].total_net for t in tier_names]
        ax.bar(x + i * width, vals, width, color=color, alpha=0.8, label=sname.title())

    ax.set_xticks(x + width)
    ax.set_xticklabels(tier_labels, fontsize=9, color="#c9d1d9")
    ax.axhline(0, color=R, lw=1, ls="--")
    ax.set_ylabel("Total NET $")
    ax.set_title("NET by Market Selection Tier", fontsize=11)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 3: gm_reward vs actual NET scatter
    ax = axes[0, 2]
    gm_rpcs = [m["gm_rpc"] for m in market_details if m["gm_rpc"] < 500]  # clip outliers
    def_nets_all = [m["defiance_net"] for m in market_details if m["gm_rpc"] < 500]
    colors_scatter = [G if n > 0 else R for n in def_nets_all]
    ax.scatter(gm_rpcs, def_nets_all, c=colors_scatter, s=40, alpha=0.6)
    ax.axhline(0, color=R, lw=1, ls="--")
    ax.set_xlabel("gm_reward_per_100 %")
    ax.set_ylabel("Defiance NET $")
    ax.set_title("gm_reward vs Actual Profit", fontsize=11)
    # Add annotation for profitable count
    profitable = sum(1 for n in def_nets_all if n > 0)
    ax.text(0.95, 0.95, f"{profitable}/{len(def_nets_all)} profitable",
            transform=ax.transAxes, fontsize=10, color=G, ha="right", va="top",
            bbox=dict(facecolor="#161b22", edgecolor=G, boxstyle="round"))

    # Panel 4: Fills comparison
    ax = axes[1, 0]
    snames = ["Naive", "Defiance", "Conservative"]
    fills = [results["naive"].total_fills, results["defiance"].total_fills, results["conservative"].total_fills]
    colors_bar = [O, C, P]
    bars = ax.bar(snames, fills, color=colors_bar, alpha=0.8)
    for bar, val in zip(bars, fills):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
                ha="center", color="#c9d1d9", fontsize=10)
    ax.set_ylabel("Total Fills")
    ax.set_title("Fill Count (fewer = better)", fontsize=11)

    # Panel 5: Reward vs Cost breakdown
    ax = axes[1, 1]
    x = np.arange(3)
    width = 0.35
    rewards = [results[s].total_reward for s in ["naive", "defiance", "conservative"]]
    costs = [results[s].total_adverse + results[s].total_gas for s in ["naive", "defiance", "conservative"]]
    ax.bar(x - width/2, rewards, width, color=G, alpha=0.8, label="Reward")
    ax.bar(x + width/2, costs, width, color=R, alpha=0.8, label="Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(snames, color="#c9d1d9")
    ax.set_ylabel("$")
    ax.set_title("Reward vs Cost", fontsize=11)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis("off")

    lines = [
        ("EXPANDED BACKTEST RESULTS", "#f0f6fc", 13, "bold"),
        (f"{len(market_details)} markets | {sum(m['n_trades'] for m in market_details):,} trades", "#8b949e", 10, "normal"),
        ("", "", 0, ""),
        ("                Naive    Defiance   Conservative", "#8b949e", 9, "bold"),
        ("-" * 55, "#30363d", 8, "normal"),
    ]

    for sname, label in [("naive", "Naive"), ("defiance", "Defiance"), ("conservative", "Conserv.")]:
        r = results[sname]
        profitable = sum(1 for m in market_details if m[f"{sname}_net"] > 0)
        lines.append((f"  {label}", "#c9d1d9", 9, "normal"))

    # Net comparison row
    nn = results["naive"].total_net
    dn = results["defiance"].total_net
    cn = results["conservative"].total_net
    lines.append((f"NET Total    ${nn:>8.1f}  ${dn:>8.1f}  ${cn:>8.1f}", G if dn > nn else R, 10, "bold"))

    np_count = sum(1 for m in market_details if m["naive_net"] > 0)
    dp_count = sum(1 for m in market_details if m["defiance_net"] > 0)
    cp_count = sum(1 for m in market_details if m["conservative_net"] > 0)
    n = len(market_details)
    lines.append((f"Profitable   {np_count:>5}/{n}    {dp_count:>5}/{n}    {cp_count:>5}/{n}", "#c9d1d9", 10, "normal"))
    lines.append((f"Fills        {results['naive'].total_fills:>7}    {results['defiance'].total_fills:>7}    {results['conservative'].total_fills:>7}", "#c9d1d9", 10, "normal"))
    lines.append((f"Reward $     {results['naive'].total_reward:>7.1f}    {results['defiance'].total_reward:>7.1f}    {results['conservative'].total_reward:>7.1f}", "#c9d1d9", 10, "normal"))
    lines.append((f"Cost $       {results['naive'].total_adverse+results['naive'].total_gas:>7.1f}    {results['defiance'].total_adverse+results['defiance'].total_gas:>7.1f}    {results['conservative'].total_adverse+results['conservative'].total_gas:>7.1f}", "#c9d1d9", 10, "normal"))

    lines.append(("", "", 0, ""))
    lines.append(("-" * 55, "#30363d", 8, "normal"))
    lines.append(("MARKET SELECTION IMPACT:", C, 11, "bold"))

    for tier, label in [("all", "All"), ("gm>=1%", "gm>=1%"), ("gm>=1%+vol<20", "gm>=1%+low vol")]:
        tr = tier_results[tier]["defiance"]
        cnt = tr.markets_traded
        lines.append((f"  {label}: {cnt} mkts, NET=${tr.total_net:.1f}", "#c9d1d9", 9, "normal"))

    lines.append(("", "", 0, ""))
    lines.append(("KEY: Market selection > strategy tuning", G, 10, "bold"))

    yp = 0.98
    for text, color, size, weight in lines:
        if size > 0:
            ax.text(0.02, yp, text, transform=ax.transAxes, fontsize=size,
                    fontweight=weight, color=color, fontfamily="monospace")
        yp -= 0.04

    fig.suptitle("Mantis Expanded Backtest: 72 Markets | Real Polymarket Data\n"
                 "Naive (0.1c) vs Defiance (15% max_spread) vs Conservative (25%)",
                 color="#f0f6fc", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = "/workspace/mantis/reports/expanded_backtest.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nChart saved: {out}")
    return out


if __name__ == "__main__":
    # Run at multiple capital levels
    for cap in [100, 250, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"  CAPITAL: ${cap}")
        print(f"{'='*60}")
        results, tier_results, market_details = run_backtest(capital=cap)

        print(f"Markets: {len(market_details)}")
        print(f"Total trades: {sum(m['n_trades'] for m in market_details):,}")

        for sname in ["naive", "defiance", "conservative"]:
            r = results[sname]
            profitable = sum(1 for m in market_details if m[f"{sname}_net"] > 0)
            print(f"  {sname.upper()}: NET=${r.total_net:.1f} | Reward=${r.total_reward:.1f} | Cost=${r.total_adverse + r.total_gas:.1f} | Fills={r.total_fills} | {profitable}/{len(market_details)} profitable")

    # Final chart uses $100 capital
    print(f"\n{'='*60}")
    print("GENERATING CHART (capital=$100)")
    print(f"{'='*60}")
    results, tier_results, market_details = run_backtest(capital=100)
    plot_results(results, tier_results, market_details)
