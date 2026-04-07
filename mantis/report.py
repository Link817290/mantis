"""Daily scan report generator."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .polymarket_client import PolymarketClient
from .scanner import MarketScanner

REPORT_DIR = Path("/workspace/mantis/reports")


def generate_report(config_path: str = "config.yaml") -> str:
    """Run a full scan and generate a plain-text daily report.

    Returns the path to the generated report file.
    """
    config = load_config(config_path)
    client = PolymarketClient()

    try:
        scanner = MarketScanner(client, config)
        start = time.time()
        results = scanner.scan()
        elapsed = time.time() - start
    finally:
        client.close()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M UTC")

    lines = []
    lines.append(f"Mantis Daily Scan Report")
    lines.append(f"Date: {date_str}  Time: {time_str}")
    lines.append(f"Scan duration: {elapsed:.0f}s")
    lines.append(f"Capital: ${config.capital}")
    lines.append(f"Viable markets found: {len(results)}")
    lines.append("")

    if not results:
        lines.append("No viable markets found.")
    else:
        lines.append(f"Top {min(10, len(results))} markets:")
        lines.append("")
        is_reward = config.engine.reward_mode
        if is_reward:
            lines.append(f"Mode: REWARD FARMING (tight spread, min_size)")
            lines.append(f"{'#':>3}  {'Est $/d':>8}  {'Pool':>7}  {'$/cap':>7}  {'Cap $':>7}  {'Q':>6}  Market")
        else:
            lines.append(f"{'#':>3}  {'Est $/day':>9}  {'Pool $/d':>8}  {'$/Q':>7}  {'Spread':>7}  {'Mid':>5}  Market")
        lines.append("-" * 95)

        for i, r in enumerate(results[:10]):
            if is_reward:
                lines.append(
                    f"{i+1:>3}  "
                    f"${r.estimated_daily_reward:>6.2f}  "
                    f"${r.market.rewards.daily_rate:>5.1f}  "
                    f"{r.reward_per_capital:>7.2f}  "
                    f"${r.capital_needed:>5.1f}  "
                    f"{r.total_q_min:>6.0f}  "
                    f"{r.market.question[:45]}"
                )
            else:
                lines.append(
                    f"{i+1:>3}  "
                    f"${r.estimated_daily_reward:>7.2f}  "
                    f"${r.market.rewards.daily_rate:>6.1f}  "
                    f"{r.reward_per_q:>7.4f}  "
                    f"{r.spread_cents:>5.1f}c  "
                    f"{r.orderbook.midpoint:>5.2f}  "
                    f"{r.market.question[:50]}"
                )

        lines.append("")
        lines.append("Details (Top 5):")
        lines.append("")

        for i, r in enumerate(results[:5]):
            ob = r.orderbook
            bid_depth = sum(l.size for l in ob.bids[:5])
            ask_depth = sum(l.size for l in ob.asks[:5])
            lines.append(f"  #{i+1}: {r.market.question[:60]}")
            lines.append(f"      Mid: {ob.midpoint:.3f}  Spread: {r.spread_cents:.1f}c")
            lines.append(f"      Reward pool: ${r.market.rewards.daily_rate}/day  Max spread: {r.market.rewards.max_spread}c")
            lines.append(f"      Min size: {r.market.rewards.min_size}  Total Q: {r.total_q_min:.0f}")
            lines.append(f"      $/Q: {r.reward_per_q:.4f}  Est daily: ${r.estimated_daily_reward:.2f}")
            lines.append(f"      Bid depth (5 lvl): {bid_depth:.0f}  Ask depth (5 lvl): {ask_depth:.0f}")
            lines.append(f"      Your depth share: {r.your_depth_share:.1%}")
            if is_reward:
                lines.append(f"      Capital needed: ${r.capital_needed:.1f}  $/capital: {r.reward_per_capital:.2f}")
            lines.append("")

        # Summary stats
        avg_reward = sum(r.estimated_daily_reward for r in results[:config.markets.max_active])
        lines.append(f"Estimated daily income (top {config.markets.max_active} markets): ${avg_reward:.2f}")

    report_text = "\n".join(lines)

    # Save to file
    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"scan_{date_str}.txt"
    report_path.write_text(report_text)

    return str(report_path)


if __name__ == "__main__":
    import os
    os.chdir("/workspace/mantis")
    path = generate_report()
    print(f"Report saved to: {path}")
    print()
    print(Path(path).read_text())
