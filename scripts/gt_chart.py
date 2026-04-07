"""Generate comparison chart: Naive vs Game-Theory reward farming."""
import sys
sys.path.insert(0, "/workspace/mantis")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mantis.backtest_gt import Config, run_comparative_backtest, run_monte_carlo

print("Running backtests...")

# Stress scenario
cfg = Config(
    daily_volatility=0.025, n_competitors=3, competitor_dist_c=0.3,
    pool_per_day=60.0, n_days=90, n_markets=2,
)
single = run_comparative_backtest(cfg)
print(f"  Single done: Naive ${single['naive']['total_net']:.0f} GT ${single['gt']['total_net']:.0f}")

# MC with 20 sims (fast)
mc = run_monte_carlo(cfg, n_sims=20)
print("  MC done")

# Calm reference
cfg_calm = Config(
    daily_volatility=0.006, n_competitors=1, competitor_dist_c=0.5,
    pool_per_day=80.0, n_days=90,
)
calm = run_comparative_backtest(cfg_calm)
print("  Calm done")

n, g = single["naive"], single["gt"]
nc, gc = calm["naive"], calm["gt"]

# ── Chart: 4 panels ──
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
days = list(range(1, 91))

# Panel 1: Calm scenario
ax = axes[0, 0]
ax.plot(days, nc["cumulative"], color=O, lw=2, label=f'Naive ${nc["total_net"]:.0f}')
ax.plot(days, gc["cumulative"], color=C, lw=2, label=f'GT ${gc["total_net"]:.0f}')
ax.set_xlabel("Day"); ax.set_ylabel("Profit $")
ax.set_title("Calm (vol=0.6%, 1 comp, $80/d)", fontsize=11)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
ax.text(0.05, 0.85, f"Fills: Naive {nc['fills_per_day']:.1f}/d  GT {gc['fills_per_day']:.1f}/d",
        transform=ax.transAxes, fontsize=9, color="#8b949e",
        bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round"))

# Panel 2: Stress scenario
ax = axes[0, 1]
ax.plot(days, n["cumulative"], color=O, lw=2.5, label=f'Naive ${n["total_net"]:.0f}')
ax.plot(days, g["cumulative"], color=C, lw=2.5, label=f'GT ${g["total_net"]:.0f}')
ax.axhline(0, color=R, lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Day"); ax.set_ylabel("Profit $")
ax.set_title("Stress (vol=2.5%, 3 comp, $60/d)", fontsize=11)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
ax.text(0.05, 0.85, f"Fills: Naive {n['fills_per_day']:.1f}/d  GT {g['fills_per_day']:.1f}/d",
        transform=ax.transAxes, fontsize=9, color="#8b949e",
        bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round"))

# Panel 3: MC fan
ax = axes[1, 0]
na = np.array(mc["naive_curves"])
ga = np.array(mc["gt_curves"])
p5n, p50n, p95n = [np.percentile(na, p, axis=0) for p in [5, 50, 95]]
p5g, p50g, p95g = [np.percentile(ga, p, axis=0) for p in [5, 50, 95]]
ax.fill_between(days, p5n, p95n, color=O, alpha=0.12)
ax.plot(days, p50n, color=O, lw=2, label=f"Naive med ${p50n[-1]:.0f}")
ax.fill_between(days, p5g, p95g, color=C, alpha=0.12)
ax.plot(days, p50g, color=C, lw=2, label=f"GT med ${p50g[-1]:.0f}")
ax.axhline(0, color=R, lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Day"); ax.set_ylabel("Profit $")
ax.set_title(f"Monte Carlo {len(mc['naive_curves'])} Sims", fontsize=11)
ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

ndd = np.median(mc["naive_dds"])
gdd = np.median(mc["gt_dds"])
ddr = (1 - gdd/max(ndd, 0.01))*100 if ndd > 0.01 else 0
ax.text(0.05, 0.05, f"DD median: N${ndd:.1f} vs GT${gdd:.1f} ({ddr:.0f}% less)",
        transform=ax.transAxes, fontsize=9, color=G, fontweight="bold",
        bbox=dict(facecolor="#161b22", edgecolor=G, boxstyle="round"))

# Panel 4: Summary
ax = axes[1, 1]
ax.axis("off")

def row(y, label, nv, gv, fmt="$.2f", better="higher"):
    ax.text(0.02, y, label, transform=ax.transAxes, fontsize=10, color="#c9d1d9", fontfamily="monospace")
    ax.text(0.48, y, f"{nv:{fmt[1:]}}" if fmt[0]=="$" else f"{nv:{fmt}}", transform=ax.transAxes,
            fontsize=10, color=O, fontfamily="monospace", ha="right")
    is_better = gv > nv if better == "higher" else gv < nv
    ax.text(0.72, y, f"{gv:{fmt[1:]}}" if fmt[0]=="$" else f"{gv:{fmt}}", transform=ax.transAxes,
            fontsize=10, color=G if is_better else R, fontfamily="monospace", fontweight="bold", ha="right")

y0 = 0.95
ax.text(0.02, y0, "STRESS RESULTS", transform=ax.transAxes, fontsize=12, color="#f0f6fc", fontweight="bold", fontfamily="monospace")
ax.text(0.28, y0-0.05, "Naive", transform=ax.transAxes, fontsize=10, color=O, fontfamily="monospace", fontweight="bold", ha="right")
ax.text(0.52, y0-0.05, "Game Theory", transform=ax.transAxes, fontsize=10, color=C, fontfamily="monospace", fontweight="bold", ha="right")

y = y0 - 0.12
row(y, "NET $/day", n["avg_net"], g["avg_net"], "$7.2f"); y -= 0.06
row(y, "90d Total", n["total_net"], g["total_net"], "$7.0f"); y -= 0.06
row(y, "Fills/day", n["fills_per_day"], g["fills_per_day"], "$7.1f", "lower"); y -= 0.06
row(y, "Avg Loss $", n["avg_loss"], g["avg_loss"], "$7.2f", "lower"); y -= 0.06
row(y, "Max DD $", n["max_drawdown"], g["max_drawdown"], "$7.1f", "lower"); y -= 0.06
row(y, "Win Rate", n["win_rate"]*100, g["win_rate"]*100, "$6.0f"); y -= 0.06
row(y, "Sharpe", n["sharpe"], g["sharpe"], "$7.1f"); y -= 0.08

ax.text(0.02, y, "CALM REFERENCE", transform=ax.transAxes, fontsize=11, color="#f0f6fc", fontweight="bold", fontfamily="monospace"); y -= 0.06
ax.text(0.02, y, f"Naive ${nc['avg_net']:.2f}/d  GT ${gc['avg_net']:.2f}/d", transform=ax.transAxes,
        fontsize=10, color="#8b949e", fontfamily="monospace"); y -= 0.08

ax.text(0.02, y, "GT LAYERS:", transform=ax.transAxes, fontsize=11, color=C, fontweight="bold", fontfamily="monospace"); y -= 0.05
for layer in ["Markov drift -> avoid adverse fills",
              "Glosten-Milgrom -> dynamic min spread",
              "Vol scaling -> widen in turbulence",
              "Post-fill widening -> reduce clusters"]:
    ax.text(0.04, y, layer, transform=ax.transAxes, fontsize=8, color="#8b949e", fontfamily="monospace")
    y -= 0.04

ax.set_title("Strategy Comparison", fontsize=11)

fig.suptitle("Mantis: Naive (0.1c fixed) vs Game Theory (dynamic spread)",
             color="#f0f6fc", fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = "/workspace/mantis/reports/gt_backtest_comparison.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"\nChart: {out}")
