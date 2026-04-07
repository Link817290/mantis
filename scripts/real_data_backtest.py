"""Real-data backtest: replay actual trades against our hypothetical orders.

Logic:
- Use real orderbook snapshots for Q calculation and competition assessment
- Use real trade history to determine when we'd get filled
- A real trade at price <= our_bid means our bid got lifted
- A real trade at price >= our_ask means our ask got lifted
- Track inventory, adverse selection, and reward accumulation
"""
import sys
sys.path.insert(0, "/workspace/mantis")

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class MarketResult:
    question: str
    condition_id: str
    daily_rate: float
    max_spread: float
    n_trades: int
    avg_q: float
    avg_spread_c: float

    # Strategy results
    naive_fills: int = 0
    naive_fill_cost: float = 0.0
    naive_reward: float = 0.0
    naive_net: float = 0.0

    gt_fills: int = 0
    gt_fill_cost: float = 0.0
    gt_reward: float = 0.0
    gt_net: float = 0.0


def q_score(size, dist_c, max_spread_c):
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


def run_real_backtest():
    conn = sqlite3.connect("/workspace/mantis/data/snapshots.db")

    # Get all markets
    markets = conn.execute("""
        SELECT m.question, m.condition_id, m.daily_rate, m.max_spread, m.token_id
        FROM markets m ORDER BY m.daily_rate DESC
    """).fetchall()

    results = []
    all_naive_daily = {}  # condition_id -> [daily_net, ...]
    all_gt_daily = {}

    for question, cid, daily_rate, max_spread, token_id in markets:
        # Get trades sorted by time
        trades = conn.execute("""
            SELECT side, price, size, timestamp FROM trades
            WHERE condition_id = ? ORDER BY timestamp ASC
        """, (cid,)).fetchall()

        if len(trades) < 10:
            continue

        # Get snapshots for this market
        snaps = conn.execute("""
            SELECT ts_unix, best_bid, best_ask, midpoint, spread_cents, total_q,
                   bids_json, asks_json
            FROM snapshots WHERE condition_id = ? ORDER BY ts_unix ASC
        """, (cid,)).fetchall()

        if not snaps:
            continue

        # Trade time range
        t_start = trades[0][3]
        t_end = trades[-1][3]
        n_days = max((t_end - t_start) / 86400, 0.1)

        # Average mid from snapshots
        avg_mid = statistics.mean(s[3] for s in snaps if s[3] and s[3] > 0)
        avg_q = statistics.mean(s[5] for s in snaps if s[5] is not None)
        avg_spread_c = statistics.mean(s[4] for s in snaps if s[4] is not None)

        min_size = 50  # standard min_size

        # === Simulate both strategies ===
        for strategy in ["naive", "gt"]:
            fills = 0
            fill_cost = 0.0

            # GT state
            recent_prices = []
            recent_fill_times = []
            current_half_c = 0.1 if strategy == "naive" else 0.3

            # Order state: can only fill once per side, then 60s cooldown
            bid_live = True
            ask_live = True
            bid_cooldown_until = 0
            ask_cooldown_until = 0
            COOLDOWN = 60  # 60 seconds to re-place after fill

            snap_idx = 0

            for side, price, size, ts in trades:
                # Check cooldowns
                if not bid_live and ts >= bid_cooldown_until:
                    bid_live = True
                if not ask_live and ts >= ask_cooldown_until:
                    ask_live = True

                # Update snap index
                while snap_idx < len(snaps) - 1 and snaps[snap_idx + 1][0] <= ts:
                    snap_idx += 1

                snap = snaps[min(snap_idx, len(snaps) - 1)]
                snap_mid = snap[3] if snap[3] and snap[3] > 0 else avg_mid

                # GT: dynamic spread
                if strategy == "gt":
                    recent_prices.append(price)
                    if len(recent_prices) > 30:
                        recent_prices = recent_prices[-30:]

                    if len(recent_prices) >= 5:
                        returns = [abs(recent_prices[i] - recent_prices[i-1])
                                  for i in range(1, len(recent_prices))]
                        vol = statistics.mean(returns) * 100
                        vol_factor = max(0.5, min(3.0, vol / 0.3))

                        recent_fill_times = [t for t in recent_fill_times if ts - t < 3600]
                        fill_rate = len(recent_fill_times)
                        fill_factor = max(1.0, fill_rate / 3.0)

                        target = max(0.2 * vol_factor * fill_factor, 0.1)
                        target = min(target, max_spread * 0.4)
                        current_half_c = current_half_c * 0.8 + target * 0.2

                half_c = 0.1 if strategy == "naive" else current_half_c
                half = half_c / 100

                our_bid = snap_mid - half
                our_ask = snap_mid + half

                # Fill check - only if that side is live
                filled = False
                if side == "SELL" and price <= our_bid and price > 0 and bid_live:
                    filled = True
                    # Adverse = spread we gave up + price gap
                    adverse = half * min_size  # we bought at bid, mid is half above
                    adverse += max(0, our_bid - price) * min_size
                    fill_cost += adverse
                    bid_live = False
                    bid_cooldown_until = ts + COOLDOWN

                elif side == "BUY" and price >= our_ask and price < 1 and ask_live:
                    filled = True
                    adverse = half * min_size
                    adverse += max(0, price - our_ask) * min_size
                    fill_cost += adverse
                    ask_live = False
                    ask_cooldown_until = ts + COOLDOWN

                if filled:
                    fills += 1
                    if strategy == "gt":
                        recent_fill_times.append(ts)
                        current_half_c = min(current_half_c * 1.5, max_spread * 0.4)

            # Reward
            our_q = q_score(min_size, half_c, max_spread)
            total_q = max(avg_q, 1.0) + our_q
            our_share = our_q / total_q
            total_reward = daily_rate * n_days * our_share

            gas = fills * 0.005
            net = total_reward - fill_cost - gas

            if strategy == "naive":
                mr = MarketResult(
                    question=question, condition_id=cid, daily_rate=daily_rate,
                    max_spread=max_spread, n_trades=len(trades), avg_q=avg_q,
                    avg_spread_c=avg_spread_c,
                    naive_fills=fills, naive_fill_cost=fill_cost + gas,
                    naive_reward=total_reward, naive_net=net,
                )
            else:
                mr.gt_fills = fills
                mr.gt_fill_cost = fill_cost + gas
                mr.gt_reward = total_reward
                mr.gt_net = net

        results.append(mr)
        print(f"  {question[:45]}")
        print(f"    Naive: fills={mr.naive_fills} cost=${mr.naive_fill_cost:.2f} reward=${mr.naive_reward:.2f} NET=${mr.naive_net:.2f}")
        print(f"    GT:    fills={mr.gt_fills} cost=${mr.gt_fill_cost:.2f} reward=${mr.gt_reward:.2f} NET=${mr.gt_net:.2f}")

    conn.close()
    return results


def plot_results(results):
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

    C, O, G, R = "#00d4ff", "#ff6b35", "#3fb950", "#f85149"

    # Sort by naive net
    results.sort(key=lambda r: -r.naive_net)
    labels = [r.question[:30] for r in results]
    n_nets = [r.naive_net for r in results]
    g_nets = [r.gt_net for r in results]

    # Panel 1: NET per market comparison
    ax = axes[0, 0]
    y = range(len(results))
    ax.barh([i - 0.15 for i in y], n_nets, height=0.3, color=O, alpha=0.8, label="Naive")
    ax.barh([i + 0.15 for i in y], g_nets, height=0.3, color=C, alpha=0.8, label="GT")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color="#c9d1d9")
    ax.axvline(0, color=R, lw=1, ls="--")
    ax.set_xlabel("NET Profit $")
    ax.set_title("Real-Data NET per Market", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax.invert_yaxis()

    # Panel 2: Fills comparison
    ax = axes[0, 1]
    n_fills = [r.naive_fills for r in results]
    g_fills = [r.gt_fills for r in results]
    ax.barh([i - 0.15 for i in y], n_fills, height=0.3, color=O, alpha=0.8, label="Naive")
    ax.barh([i + 0.15 for i in y], g_fills, height=0.3, color=C, alpha=0.8, label="GT")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color="#c9d1d9")
    ax.set_xlabel("Total Fills")
    ax.set_title("Fills (fewer = better for rewards)", fontsize=12)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
    ax.invert_yaxis()

    # Panel 3: Reward vs Cost scatter
    ax = axes[1, 0]
    for r in results:
        ax.scatter(r.naive_reward, r.naive_fill_cost, color=O, s=80, alpha=0.7, zorder=5)
        ax.scatter(r.gt_reward, r.gt_fill_cost, color=C, s=80, alpha=0.7, marker="^", zorder=5)
        # Draw arrow from naive to GT
        ax.annotate("", xy=(r.gt_reward, r.gt_fill_cost),
                    xytext=(r.naive_reward, r.naive_fill_cost),
                    arrowprops=dict(arrowstyle="->", color="#8b949e", alpha=0.3))

    max_val = max(max(r.naive_reward for r in results), max(r.naive_fill_cost for r in results)) * 1.1
    ax.plot([0, max_val], [0, max_val], color=R, ls="--", alpha=0.5, label="Break-even")
    ax.set_xlabel("Reward $")
    ax.set_ylabel("Fill Cost + Inventory Risk $")
    ax.set_title("Reward vs Cost (below line = profit)", fontsize=12)
    ax.scatter([], [], color=O, s=60, label="Naive (circle)")
    ax.scatter([], [], color=C, s=60, marker="^", label="GT (triangle)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 4: Summary
    ax = axes[1, 1]
    ax.axis("off")

    total_naive = sum(r.naive_net for r in results)
    total_gt = sum(r.gt_net for r in results)
    naive_profitable = sum(1 for r in results if r.naive_net > 0)
    gt_profitable = sum(1 for r in results if r.gt_net > 0)
    total_naive_fills = sum(r.naive_fills for r in results)
    total_gt_fills = sum(r.gt_fills for r in results)

    # Best markets
    best_naive = max(results, key=lambda r: r.naive_net)
    best_gt = max(results, key=lambda r: r.gt_net)
    worst = min(results, key=lambda r: min(r.naive_net, r.gt_net))

    lines = [
        ("REAL DATA BACKTEST RESULTS", "#f0f6fc", 14, "bold"),
        (f"10 markets | {sum(r.n_trades for r in results)} real trades", "#8b949e", 10, "normal"),
        ("", "", 0, ""),
        ("                  Naive     GT       Delta", "#8b949e", 10, "bold"),
        ("-" * 50, "#30363d", 9, "normal"),
        (f"Total NET     ${total_naive:>8.1f}  ${total_gt:>8.1f}  {(total_gt-total_naive)/max(abs(total_naive),1)*100:>+5.0f}%",
         G if total_gt > total_naive else R, 10, "normal"),
        (f"Profitable     {naive_profitable:>5}/{len(results)}    {gt_profitable:>5}/{len(results)}",
         "#c9d1d9", 10, "normal"),
        (f"Total Fills    {total_naive_fills:>6}    {total_gt_fills:>6}    {(total_gt_fills-total_naive_fills)/max(total_naive_fills,1)*100:>+5.0f}%",
         G if total_gt_fills < total_naive_fills else R, 10, "normal"),
        ("", "", 0, ""),
        ("-" * 50, "#30363d", 9, "normal"),
        (f"Best:  {best_gt.question[:35]}", C, 9, "bold"),
        (f"  N=${best_naive.naive_net:.1f}  GT=${best_gt.gt_net:.1f}", "#c9d1d9", 9, "normal"),
        (f"Worst: {worst.question[:35]}", R, 9, "bold"),
        (f"  N=${worst.naive_net:.1f}  GT=${worst.gt_net:.1f}", "#c9d1d9", 9, "normal"),
        ("", "", 0, ""),
        ("-" * 50, "#30363d", 9, "normal"),
        ("KEY INSIGHT:", "#f0f6fc", 11, "bold"),
        ("GT reduces fills via:", C, 10, "normal"),
        ("  - Vol-based spread widening", "#8b949e", 9, "normal"),
        ("  - Fill-rate adaptive retreat", "#8b949e", 9, "normal"),
        ("  - Post-fill defensive widening", "#8b949e", 9, "normal"),
        ("Tradeoff: fewer fills but less Q", O, 9, "normal"),
    ]

    yp = 0.98
    for text, color, size, weight in lines:
        if size > 0:
            ax.text(0.02, yp, text, transform=ax.transAxes, fontsize=size,
                    fontweight=weight, color=color, fontfamily="monospace")
        yp -= 0.04

    fig.suptitle("Mantis Real-Data Backtest: Naive vs Game Theory\n"
                 "Based on actual Polymarket trades & orderbook snapshots",
                 color="#f0f6fc", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = "/workspace/mantis/reports/real_data_gt_comparison.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nChart saved: {out}")
    return out


if __name__ == "__main__":
    print("=== Real-Data Backtest: Naive vs Game Theory ===\n")
    results = run_real_backtest()
    print(f"\n=== TOTALS ===")
    tn = sum(r.naive_net for r in results)
    tg = sum(r.gt_net for r in results)
    print(f"Naive total NET: ${tn:.1f}")
    print(f"GT total NET:    ${tg:.1f}")
    print(f"Naive fills: {sum(r.naive_fills for r in results)}")
    print(f"GT fills:    {sum(r.gt_fills for r in results)}")
    plot_results(results)
