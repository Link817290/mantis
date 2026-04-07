"""Backtest: Naive vs Game-Theory Enhanced Reward Farming.

Compares two strategies on the same price paths:
  1. Naive: fixed 0.1c from mid, always on
  2. Game Theory: Markov drift + GM min spread + ecosystem detection + dynamic pullback

Uses calibrated parameters from real Polymarket data.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field


def q_score(size: float, dist_c: float, max_spread_c: float) -> float:
    if dist_c >= max_spread_c:
        return 0.0
    return ((max_spread_c - dist_c) / max_spread_c) ** 2 * size


@dataclass
class Config:
    capital: float = 100.0
    n_markets: int = 2
    min_size: int = 50
    max_spread_c: float = 4.5
    pool_per_day: float = 80.0
    tick_seconds: int = 300  # 5-min ticks for speed
    daily_volatility: float = 0.006  # calibrated 0.6%
    n_competitors: int = 1
    competitor_size: int = 50
    competitor_dist_c: float = 0.5
    n_days: int = 90
    seed: int = 42


# ── Markov drift estimator ──
def estimate_drift(prices: list[float], bins: int = 5) -> float:
    if len(prices) < 5:
        return 0.0
    min_p, max_p = min(prices), max(prices)
    rng = max_p - min_p
    if rng < 0.0005:
        return 0.0
    bs = rng / bins

    def to_bin(p):
        return min(int((p - min_p) / bs), bins - 1)

    def center(b):
        return min_p + (b + 0.5) * bs

    trans = {}
    for i in range(len(prices) - 1):
        f, t = to_bin(prices[i]), to_bin(prices[i + 1])
        trans.setdefault(f, {})
        trans[f][t] = trans[f].get(t, 0) + 1

    cur = to_bin(prices[-1])
    row = trans.get(cur, {})
    total = sum(row.values())
    if total == 0:
        return 0.0
    expected = sum(center(b) * c / total for b, c in row.items())
    return expected - center(cur)


# ── GM adverse selection detector ──
def gm_min_half_spread_c(recent_fills: list[float], base: float = 0.1) -> float:
    """Returns minimum safe half-spread in cents based on recent adverse selection."""
    if len(recent_fills) < 3:
        return base
    # Measure average fill-time adverse move (in cents)
    avg_adverse = statistics.mean(recent_fills) * 100
    # GM formula: min_spread proportional to informed fraction * expected jump
    # We use avg_adverse as a proxy
    return max(base, avg_adverse * 2.0)


def run_comparative_backtest(cfg: Config):
    """Run both naive and GT strategies on identical price paths.

    Returns dict with daily series for both strategies.
    """
    random.seed(cfg.seed)
    ticks_per_day = 86400 // cfg.tick_seconds
    tick_vol = cfg.daily_volatility / math.sqrt(ticks_per_day)

    comp_q = q_score(cfg.competitor_size, cfg.competitor_dist_c, cfg.max_spread_c) * cfg.n_competitors

    # Pre-generate all price paths so both strategies see exactly the same prices
    all_paths = []
    all_shocks = []
    for day in range(cfg.n_days):
        day_paths = []
        day_shocks = []
        for m in range(cfg.n_markets):
            mid0 = 0.40 + random.random() * 0.20
            mids = [mid0]
            shocks = []
            for t in range(ticks_per_day):
                s = random.gauss(0, tick_vol)
                shocks.append(s)
                mids.append(max(0.02, min(0.98, mids[-1] + s)))
            day_paths.append(mids)
            day_shocks.append(shocks)
        all_paths.append(day_paths)
        all_shocks.append(day_shocks)

    def simulate(strategy: str):
        daily_rewards = []
        daily_losses = []
        daily_nets = []
        cumulative_nets = []
        total_fills = 0
        cumulative = 0.0

        for day in range(cfg.n_days):
            day_q_ticks = 0
            day_total_ticks = 0
            day_fills = 0
            day_adverse = 0.0

            for mi in range(cfg.n_markets):
                mids = all_paths[day][mi]
                shocks = all_shocks[day][mi]

                bid_live = True
                ask_live = True
                cooldown = 0

                # GT state
                price_window = []
                recent_adverse_moves = []
                vol_window = []
                current_half_c = 0.1 if strategy == "naive" else 0.3  # GT starts wider

                for t in range(ticks_per_day):
                    day_total_ticks += 1
                    mid = mids[t + 1]
                    shock = shocks[t]

                    # GT: track state
                    if strategy == "gt":
                        price_window.append(mid)
                        if len(price_window) > 20:
                            price_window = price_window[-20:]
                        vol_window.append(abs(shock))
                        if len(vol_window) > 30:
                            vol_window = vol_window[-30:]

                    if cooldown > 0:
                        cooldown -= 1
                        if cooldown == 0:
                            bid_live = True
                            ask_live = True
                            # GT: after fill, recalculate spread
                            if strategy == "gt":
                                # Widen after fill (defensive)
                                current_half_c = min(current_half_c * 1.5, cfg.max_spread_c * 0.8)
                        continue

                    # GT: dynamic spread adjustment every 60 ticks (~30 min)
                    if strategy == "gt" and t % 6 == 0 and t > 0:
                        # Layer 1: Markov drift
                        drift = estimate_drift(price_window)
                        drift_c = abs(drift) * 100

                        # Layer 2: GM min spread from adverse selection
                        gm_min = gm_min_half_spread_c(recent_adverse_moves, base=0.1)

                        # Layer 3: Realized vol scaling
                        if len(vol_window) >= 10:
                            realized_vol = statistics.mean(vol_window[-10:])
                            expected_vol = tick_vol
                            vol_ratio = realized_vol / expected_vol if expected_vol > 0 else 1.0
                            # High vol → widen, low vol → tighten
                            vol_factor = max(0.5, min(3.0, vol_ratio))
                        else:
                            vol_factor = 1.0

                        # Combine: base 0.2c, scaled by vol, floored by GM
                        target = max(0.2 * vol_factor, gm_min, drift_c * 1.5)
                        # Clamp
                        target = max(0.1, min(target, cfg.max_spread_c * 0.5))

                        # Smooth transition (don't jump)
                        current_half_c = current_half_c * 0.7 + target * 0.3

                        # Layer 4: Strong drift → pull one side
                        # (handled below in fill logic via asymmetric placement)

                    # Effective half-spread for this tick
                    if strategy == "naive":
                        half_c = 0.1
                    else:
                        half_c = current_half_c

                    half = half_c / 100

                    # Q accumulation
                    if bid_live and ask_live:
                        our_q = q_score(cfg.min_size, half_c, cfg.max_spread_c)
                        day_q_ticks += 1  # simplified: count ticks

                    # Fill probability
                    fill_prob = 0.007 * math.exp(-half_c / 0.15)

                    # GT: Markov-adjusted fill avoidance
                    if strategy == "gt" and len(price_window) >= 5:
                        drift = estimate_drift(price_window[-10:])
                        # If strong downward drift and bid is live, temporarily pull bid
                        if drift < -0.0005 and bid_live:
                            # Reduce fill prob by pulling back (effectively widening)
                            fill_prob *= 0.3
                        if drift > 0.0005 and ask_live:
                            fill_prob *= 0.3

                    # Bid fill
                    if shock < 0 and bid_live:
                        adj_prob = fill_prob + abs(shock) * 30
                        if random.random() < adj_prob:
                            adverse = abs(shock) * cfg.min_size
                            adverse += half * cfg.min_size * 0.3
                            day_adverse += adverse
                            bid_live = False
                            cooldown = 1
                            day_fills += 1
                            if strategy == "gt":
                                recent_adverse_moves.append(abs(shock))
                                if len(recent_adverse_moves) > 20:
                                    recent_adverse_moves = recent_adverse_moves[-20:]

                    # Ask fill
                    if shock > 0 and ask_live:
                        adj_prob = fill_prob + abs(shock) * 30
                        if random.random() < adj_prob:
                            adverse = abs(shock) * cfg.min_size
                            adverse += half * cfg.min_size * 0.3
                            day_adverse += adverse
                            ask_live = False
                            cooldown = 1
                            day_fills += 1
                            if strategy == "gt":
                                recent_adverse_moves.append(abs(shock))
                                if len(recent_adverse_moves) > 20:
                                    recent_adverse_moves = recent_adverse_moves[-20:]

            # Daily reward - use average Q over day
            # Naive: fixed Q at 0.1c
            if strategy == "naive":
                our_q_val = q_score(cfg.min_size, 0.1, cfg.max_spread_c)
            else:
                our_q_val = q_score(cfg.min_size, current_half_c, cfg.max_spread_c)

            uptime = day_q_ticks / day_total_ticks if day_total_ticks > 0 else 0
            eff_q = our_q_val * uptime
            total_q = eff_q + comp_q + 1
            share = eff_q / total_q if total_q > 0 else 0
            day_reward = cfg.pool_per_day * share * cfg.n_markets
            gas = day_fills * 0.005

            day_net = day_reward - day_adverse - gas
            daily_rewards.append(day_reward)
            daily_losses.append(day_adverse + gas)
            daily_nets.append(day_net)
            total_fills += day_fills

            cumulative += day_net
            cumulative_nets.append(cumulative)

        # Stats
        avg_net = statistics.mean(daily_nets)
        std_net = statistics.stdev(daily_nets) if len(daily_nets) > 1 else 0.01
        sharpe = (avg_net / std_net * math.sqrt(365)) if std_net > 0 else 0

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for c in cumulative_nets:
            if c > peak:
                peak = c
            dd = peak - c
            if dd > max_dd:
                max_dd = dd

        return {
            "daily_rewards": daily_rewards,
            "daily_losses": daily_losses,
            "daily_nets": daily_nets,
            "cumulative": cumulative_nets,
            "avg_reward": statistics.mean(daily_rewards),
            "avg_loss": statistics.mean(daily_losses),
            "avg_net": avg_net,
            "std_net": std_net,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "max_dd_pct": max_dd / cfg.capital,
            "total_fills": total_fills,
            "fills_per_day": total_fills / cfg.n_days,
            "win_rate": sum(1 for d in daily_nets if d > 0) / cfg.n_days,
            "total_net": sum(daily_nets),
        }

    naive = simulate("naive")
    gt = simulate("gt")

    return {"naive": naive, "gt": gt, "config": cfg}


def run_monte_carlo(cfg: Config, n_sims: int = 200):
    """Run Monte Carlo with different seeds for both strategies."""
    all_naive_cum = []
    all_gt_cum = []
    naive_dds = []
    gt_dds = []
    naive_totals = []
    gt_totals = []

    for i in range(n_sims):
        c = Config(
            capital=cfg.capital, n_markets=cfg.n_markets, min_size=cfg.min_size,
            max_spread_c=cfg.max_spread_c, pool_per_day=cfg.pool_per_day,
            daily_volatility=cfg.daily_volatility, n_competitors=cfg.n_competitors,
            competitor_size=cfg.competitor_size, competitor_dist_c=cfg.competitor_dist_c,
            n_days=cfg.n_days, seed=cfg.seed + i,
        )
        r = run_comparative_backtest(c)
        all_naive_cum.append(r["naive"]["cumulative"])
        all_gt_cum.append(r["gt"]["cumulative"])
        naive_dds.append(r["naive"]["max_drawdown"])
        gt_dds.append(r["gt"]["max_drawdown"])
        naive_totals.append(r["naive"]["total_net"])
        gt_totals.append(r["gt"]["total_net"])

    return {
        "naive_curves": all_naive_cum,
        "gt_curves": all_gt_cum,
        "naive_dds": naive_dds,
        "gt_dds": gt_dds,
        "naive_totals": naive_totals,
        "gt_totals": gt_totals,
        "n_days": cfg.n_days,
        "capital": cfg.capital,
    }


if __name__ == "__main__":
    cfg = Config()
    r = run_comparative_backtest(cfg)
    print("=== Single Run Comparison ===")
    for name in ["naive", "gt"]:
        s = r[name]
        print(f"\n{name.upper()}:")
        print(f"  Avg NET: ${s['avg_net']:.2f}/d")
        print(f"  Total:   ${s['total_net']:.0f}")
        print(f"  Fills:   {s['fills_per_day']:.1f}/d")
        print(f"  Max DD:  ${s['max_drawdown']:.1f} ({s['max_dd_pct']:.0%})")
        print(f"  Sharpe:  {s['sharpe']:.1f}")
        print(f"  Win:     {s['win_rate']:.0%}")
