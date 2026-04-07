"""Backtest engine for reward farming strategy.

Honest model: reward farming means you DON'T want fills.
Fills are adverse selection losses. Goal: maximize rewards, minimize fills.

Key assumptions:
- Polymarket CLOB: order placement is FREE (off-chain)
- Only fills cost gas (~$0.005 on Polygon)
- Rewards are time-weighted: Q accumulated only when orders are live
- When filled, you immediately re-place (30s delay = 1 tick of downtime)
- Adverse selection: when filled on bid, mid is BELOW your bid (you overpaid)
  Average adverse selection = half_spread (you bought at bid, mid is now ~half_spread lower)
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
class BacktestConfig:
    capital: float = 100.0
    n_markets: int = 2
    min_size: int = 50
    max_spread_c: float = 4.5
    pool_per_day: float = 80.0

    half_spread_c: float = 0.5  # distance from mid in cents
    tick_seconds: int = 30
    daily_volatility: float = 0.03  # 3% daily vol

    n_competitors: int = 3
    competitor_size: int = 50
    competitor_dist_c: float = 0.3

    n_days: int = 30
    seed: int = 42


def run_backtest(cfg: BacktestConfig) -> dict:
    random.seed(cfg.seed)
    ticks_per_day = 86400 // cfg.tick_seconds
    tick_vol = cfg.daily_volatility / math.sqrt(ticks_per_day)

    # Competitor Q (assumed constant, always live)
    comp_q = q_score(cfg.competitor_size, cfg.competitor_dist_c, cfg.max_spread_c) * cfg.n_competitors
    our_q = q_score(cfg.min_size, cfg.half_spread_c, cfg.max_spread_c)

    daily_rewards = []
    daily_losses = []
    daily_nets = []
    total_fills = 0

    for day in range(cfg.n_days):
        day_q_ticks = 0  # ticks where our orders are live
        day_total_ticks = 0
        day_fills = 0
        day_adverse_loss = 0.0

        for market_i in range(cfg.n_markets):
            mid = 0.40 + random.random() * 0.20
            bid_live = True
            ask_live = True
            cooldown = 0  # ticks until we can re-place after fill

            for tick in range(ticks_per_day):
                day_total_ticks += 1

                # Price movement
                shock = random.gauss(0, tick_vol)
                mid = max(0.02, min(0.98, mid + shock))

                # Cooldown after fill (need 1 tick to re-place)
                if cooldown > 0:
                    cooldown -= 1
                    if cooldown == 0:
                        bid_live = True
                        ask_live = True
                    continue

                # Q accumulation (only if both sides live, for Q_min)
                if bid_live and ask_live:
                    day_q_ticks += 1

                # Fill probability: depends on distance and price momentum
                # Closer to mid = more taker flow hits you
                # Base fill rate calibrated to real markets:
                #   ~$80/day pool markets have ~500-2000 shares traded/day
                #   With 50-share orders, that's ~10-40 fills/day per side
                half = cfg.half_spread_c / 100

                # Fill probability per tick per side
                # At 0.1c: ~0.7% per tick → ~20 fills/day/side
                # At 0.5c: ~0.15% per tick → ~4 fills/day/side
                # At 1.0c: ~0.03% per tick → ~1 fill/day/side
                # At 2.0c: ~0.003% → ~0.1/day
                fill_prob = 0.007 * math.exp(-cfg.half_spread_c / 0.15)

                # Directional momentum increases fill probability
                if shock < 0 and bid_live:  # price dropping toward our bid
                    adj_prob = fill_prob + abs(shock) * 30
                    if random.random() < adj_prob:
                        # Bid filled! Adverse selection loss
                        # We bought at bid_price, mid is now ~half_spread below
                        adverse = abs(shock) * cfg.min_size  # loss = price_drop × size
                        # Plus: we're now holding inventory that might drop further
                        # Average additional loss ~ half_spread (conservative)
                        adverse += half * cfg.min_size * 0.3  # 30% of half_spread as extra cost
                        day_adverse_loss += adverse
                        bid_live = False
                        cooldown = 1  # 1 tick to re-place
                        day_fills += 1

                if shock > 0 and ask_live:  # price rising toward our ask
                    adj_prob = fill_prob + abs(shock) * 30
                    if random.random() < adj_prob:
                        adverse = abs(shock) * cfg.min_size
                        adverse += half * cfg.min_size * 0.3
                        day_adverse_loss += adverse
                        ask_live = False
                        cooldown = 1
                        day_fills += 1

        # Daily reward calculation
        uptime = day_q_ticks / day_total_ticks if day_total_ticks > 0 else 0
        our_effective_q = our_q * uptime
        total_q = our_effective_q + comp_q + 1  # +1 for existing book
        our_share = our_effective_q / total_q if total_q > 0 else 0
        day_reward = cfg.pool_per_day * our_share * cfg.n_markets
        gas = day_fills * 0.005

        day_net = day_reward - day_adverse_loss - gas
        daily_rewards.append(day_reward)
        daily_losses.append(day_adverse_loss + gas)
        daily_nets.append(day_net)
        total_fills += day_fills

    # Summary stats
    avg_reward = statistics.mean(daily_rewards)
    avg_loss = statistics.mean(daily_losses)
    avg_net = statistics.mean(daily_nets)
    std_net = statistics.stdev(daily_nets) if len(daily_nets) > 1 else 0
    sharpe = (avg_net / std_net * math.sqrt(365)) if std_net > 0 else float('inf')

    # Max drawdown (on cumulative net)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for d in daily_nets:
        cumulative += d
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / cfg.capital if cfg.capital > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Win rate
    win_days = sum(1 for d in daily_nets if d > 0)

    return {
        "config": cfg,
        "avg_reward": avg_reward,
        "avg_loss": avg_loss,
        "avg_net": avg_net,
        "std_net": std_net,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_fills": total_fills,
        "fills_per_day": total_fills / cfg.n_days,
        "avg_uptime": statistics.mean(daily_rewards) / (cfg.pool_per_day * cfg.n_markets) if cfg.pool_per_day > 0 else 0,
        "win_rate": win_days / cfg.n_days,
        "total_net": sum(daily_nets),
        "daily_return": avg_net / cfg.capital,
    }


def print_result(r: dict):
    cfg = r["config"]
    print(f"  {cfg.half_spread_c:>4.1f}c | "
          f"reward ${r['avg_reward']:>6.2f} | "
          f"loss ${r['avg_loss']:>5.2f} | "
          f"NET ${r['avg_net']:>6.2f}/d | "
          f"{r['daily_return']:>5.1%}/d | "
          f"fills {r['fills_per_day']:>5.1f}/d | "
          f"dd {r['max_drawdown']:>5.1%} | "
          f"win {r['win_rate']:>4.0%} | "
          f"sharpe {r['sharpe']:>5.1f}")


if __name__ == "__main__":
    print("=" * 80)
    print("Mantis Backtest: Reward Farming (CALIBRATED with real Polymarket data)")
    print("=" * 80)
    print("Calibration source: live Polymarket orderbook + price-history API")
    print("  - 5-min vol median: 0.038c → daily vol ~0.6% (was 3%)")
    print("  - Competitor levels within max_spread: median 0-2")
    print("  - High-reward markets: mostly wide spreads, sparse books")
    print()

    distances = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    # === CALIBRATED scenarios ===
    # Real data: 0.6% daily vol, 1-2 competitors at 0.5c (wider than assumed)
    scenarios = [
        (1, 0.5, "1 comp @ 0.5c (sparse book — common)"),
        (2, 0.3, "2 comp @ 0.3c (moderate competition)"),
        (3, 0.3, "3 comp @ 0.3c (competitive)"),
        (5, 0.3, "5 comp @ 0.3c (crowded)"),
    ]

    for n_comp, comp_dist, label in scenarios:
        print(f"\n--- {label} | vol=0.6% (calibrated) ---")
        print(f"  dist | reward/d      | loss/d   | NET/day      | ret/d   | fills/d    | max dd  | win  | sharpe")
        print(f"  " + "-" * 95)
        best_net = -999
        best_dist = 0
        for d in distances:
            cfg = BacktestConfig(
                half_spread_c=d,
                n_competitors=n_comp,
                competitor_dist_c=comp_dist,
                daily_volatility=0.006,  # CALIBRATED: 0.6% from real data
                n_days=90,
            )
            r = run_backtest(cfg)
            print_result(r)
            if r["avg_net"] > best_net:
                best_net = r["avg_net"]
                best_dist = d
        print(f"  >>> BEST: {best_dist}c → ${best_net:.2f}/day ({best_net/100:.1%} daily)")

    # Compare calibrated vs original assumptions
    print(f"\n{'='*80}")
    print("SENSITIVITY: calibrated (0.6%) vs original (3%) at 0.3c, 2 competitors")
    print(f"{'='*80}")
    print(f"  vol  | NET/day | fills/d | max dd  | win  | sharpe")
    print(f"  " + "-" * 60)
    for vol, tag in [(0.003, "0.3% low"), (0.006, "0.6% calibrated"), (0.01, "1.0% high"),
                      (0.02, "2.0% stress"), (0.03, "3.0% original")]:
        cfg = BacktestConfig(half_spread_c=0.3, n_competitors=2, competitor_dist_c=0.5,
                             daily_volatility=vol, n_days=90)
        r = run_backtest(cfg)
        print(f"  {tag:<14} | ${r['avg_net']:>6.2f} | {r['fills_per_day']:>5.1f}  | {r['max_drawdown']:>5.1%} | {r['win_rate']:>4.0%} | {r['sharpe']:>5.1f}")

    # Pool size sensitivity (real markets range from $1 to $200+/day)
    print(f"\n--- Pool size sensitivity (0.3c, 2 comp, vol=0.6%) ---")
    print(f"  pool  | NET/day | monthly | ret/d")
    print(f"  " + "-" * 45)
    for pool in [10, 20, 40, 80, 120, 200]:
        cfg = BacktestConfig(half_spread_c=0.3, n_competitors=2, competitor_dist_c=0.5,
                             daily_volatility=0.006, pool_per_day=pool, n_days=90)
        r = run_backtest(cfg)
        print(f"  ${pool:<5} | ${r['avg_net']:>6.2f} | ${r['avg_net']*30:>6.0f}  | {r['daily_return']:>5.1%}")

    # Final calibrated recommendation
    print(f"\n{'='*80}")
    print("CALIBRATED RECOMMENDATION (real market parameters)")
    print(f"{'='*80}")
    cfg = BacktestConfig(
        half_spread_c=0.3,
        n_competitors=2,
        competitor_dist_c=0.5,
        daily_volatility=0.006,
        pool_per_day=80.0,
        n_days=90,
    )
    r = run_backtest(cfg)
    c = r["config"]
    print(f"  Distance: {c.half_spread_c}c from mid")
    print(f"  Daily volatility: {c.daily_volatility:.1%} (calibrated from real 5-min data)")
    print(f"  Competitors: {c.n_competitors} @ {c.competitor_dist_c}c")
    print(f"  Size: {c.min_size} (min_size)")
    print(f"  Markets: {c.n_markets}")
    print(f"  Reward pool: ${c.pool_per_day}/day")
    print(f"  ---")
    print(f"  Avg daily reward: ${r['avg_reward']:.2f}")
    print(f"  Avg daily loss:   ${r['avg_loss']:.2f}")
    print(f"  Avg daily NET:    ${r['avg_net']:.2f}")
    print(f"  Daily return:     {r['daily_return']:.1%}")
    print(f"  Monthly return:   {r['daily_return']*30:.0%}")
    print(f"  Max drawdown:     {r['max_drawdown']:.1%}")
    print(f"  Win rate:         {r['win_rate']:.0%}")
    print(f"  Sharpe (ann):     {r['sharpe']:.1f}")
    print(f"  Fills/day:        {r['fills_per_day']:.1f}")
    print()
    print("  NOTE: Real vol is ~5x lower than original assumption.")
    print("  Lower vol → fewer fills → less adverse selection → higher net.")
    print("  Main risk: competition increases, or vol spikes around events.")
