"""Market scanner - finds optimal markets based on reward/Q ratio."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .config import MantisConfig, EngineConfig
from .polymarket_client import PolymarketClient
from .strategy_log import slog
from .types import Market, Orderbook, ScanResult

logger = logging.getLogger("mantis.scanner")


def compute_q_score(size: float, distance_cents: float, max_spread: float) -> float:
    """Compute Q score for an order: S(v,s) = ((v - s) / v)^2 * size."""
    if distance_cents >= max_spread:
        return 0.0
    coeff = ((max_spread - distance_cents) / max_spread) ** 2
    return size * coeff


def compute_total_q_min(
    orderbook: Orderbook, max_spread: float,
) -> float:
    """Compute total Q_min from visible orderbook.

    Q_min uses the two-sided formula:
    - Mid 0.10-0.90: Q_min = max(min(Q_bid, Q_ask), max(Q_bid/3, Q_ask/3))
    - Extreme: Q_min = min(Q_bid, Q_ask)
    """
    mid = orderbook.midpoint

    total_q_bid = 0.0
    for level in orderbook.bids:
        dist = abs(mid - level.price) * 100
        total_q_bid += compute_q_score(level.size, dist, max_spread)

    total_q_ask = 0.0
    for level in orderbook.asks:
        dist = abs(level.price - mid) * 100
        total_q_ask += compute_q_score(level.size, dist, max_spread)

    if 0.10 <= mid <= 0.90:
        return max(min(total_q_bid, total_q_ask), max(total_q_bid / 3, total_q_ask / 3))
    else:
        return min(total_q_bid, total_q_ask)


def estimate_your_q(
    capital_per_side: float,
    mid: float,
    half_spread: float,
    max_spread: float,
) -> float:
    """Estimate your Q_min given capital allocation."""
    bid_price = round(mid - half_spread, 3)
    ask_price = round(mid + half_spread, 3)

    if bid_price <= 0 or ask_price >= 1:
        return 0.0

    bid_qty = int(capital_per_side / bid_price)
    ask_qty = int(capital_per_side / ask_price)

    dist_cents = half_spread * 100
    q_bid = compute_q_score(bid_qty, dist_cents, max_spread)
    q_ask = compute_q_score(ask_qty, dist_cents, max_spread)

    return min(q_bid, q_ask)


class MarketScanner:
    def __init__(self, client: PolymarketClient, config: MantisConfig):
        self.client = client
        self.cfg = config.scanner
        self.capital = config.capital
        self.reward_mode = config.engine.reward_mode
        self.reward_spread_pct = config.engine.reward_spread_pct
        self.default_half_spread = config.engine.default_half_spread
        self.max_q_competition = config.risk.max_q_competition

    def scan(self, force_include_cids: set[str] | None = None) -> list[ScanResult]:
        """Scan all markets and return ranked candidates.

        Args:
            force_include_cids: condition_ids to always include (e.g. markets with positions)
        """
        logger.info("Starting market scan...")
        start = time.time()
        force_cids = force_include_cids or set()

        all_markets = self.client.fetch_all_sampling_markets()

        # Phase 1: filter by basic criteria
        candidates = []
        for m in all_markets:
            if not m.active or not m.yes_token:
                continue

            # Always include markets with existing positions (to unwind inventory)
            if m.condition_id in force_cids:
                candidates.append(m)
                continue

            if m.rewards.daily_rate < self.cfg.min_reward_rate:
                continue
            if m.rewards.min_size > self.cfg.max_min_size:
                continue
            price = m.yes_token.price
            if not (self.cfg.min_price <= price <= self.cfg.max_price):
                continue
            # Skip markets settling within 48h
            if m.end_date:
                try:
                    end_dt = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
                    hours_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    if hours_left < 48:
                        continue
                except (ValueError, TypeError):
                    pass
            candidates.append(m)

        # Pre-sort by reward rate and limit to top 100 to avoid excessive API calls
        candidates.sort(key=lambda m: -m.rewards.daily_rate)
        if len(candidates) > 100:
            logger.info(f"Phase 1: {len(candidates)} candidates → trimming to top 100 by reward rate")
            candidates = candidates[:100]
        else:
            logger.info(f"Phase 1: {len(candidates)} candidates from {len(all_markets)} markets")

        # Phase 2: fetch orderbooks and compute reward/Q
        results: list[ScanResult] = []
        for m in candidates:
            try:
                token = m.yes_token
                assert token is not None
                ob = self.client.fetch_orderbook(token.token_id)
                is_position_market = m.condition_id in force_cids

                if not ob.bids or not ob.asks:
                    if not is_position_market:
                        continue

                # Skip markets with extreme spread (illiquid or stale)
                if ob.spread_cents > 5.0 and not is_position_market:
                    continue

                # Skip markets with thin books (no real liquidity)
                bid_depth_5 = sum(l.size for l in ob.bids[:5])
                ask_depth_5 = sum(l.size for l in ob.asks[:5])
                if (bid_depth_5 < 50 or ask_depth_5 < 50) and not is_position_market:
                    continue

                total_q = compute_total_q_min(ob, m.rewards.max_spread)
                if total_q < 1.0:
                    total_q = 1.0  # treat near-empty books as Q=1

                # Reject markets with too much Q competition (but keep position markets)
                if self.reward_mode and total_q > self.max_q_competition and not is_position_market:
                    slog.scan_skip(
                        market=m.question[:40], reason="q_competition",
                        total_q=total_q, threshold=self.max_q_competition,
                    )
                    continue

                # Estimate your Q
                if self.reward_mode:
                    # Reward mode: distance = 15% of max_spread (defiance_cr)
                    half_sp = (m.rewards.max_spread * self.reward_spread_pct) / 100
                    half_sp = max(half_sp, 0.001)
                    your_q = estimate_your_q(
                        m.rewards.min_size * ob.midpoint,  # capital = min_size * price
                        ob.midpoint, half_sp, m.rewards.max_spread,
                    )
                    cap_needed = m.rewards.min_size * ob.midpoint * 2  # both sides
                else:
                    half_sp = self.default_half_spread
                    capital_per_side = self.capital / 2
                    your_q = estimate_your_q(
                        capital_per_side, ob.midpoint, half_sp, m.rewards.max_spread,
                    )
                    cap_needed = self.capital

                reward_per_q = m.rewards.daily_rate / total_q
                # Cap reward_per_q: if you'd be the only maker, cap at full pool
                reward_per_q = min(reward_per_q, m.rewards.daily_rate)
                your_share = your_q / (total_q + your_q)
                est_daily = m.rewards.daily_rate * your_share

                # Your depth share
                bid_depth = sum(l.size for l in ob.bids[:5])
                bid_qty = m.rewards.min_size if self.reward_mode else (
                    int((self.capital / 2) / ob.midpoint) if ob.midpoint > 0 else 0
                )
                depth_share = bid_qty / (bid_depth + bid_qty) if bid_depth > 0 else 1.0

                # Reward per capital (key metric for reward mode)
                rpc = est_daily / cap_needed if cap_needed > 0 else 0

                slog.scan_result(
                    market=m.question[:40], condition_id=m.condition_id,
                    daily_rate=m.rewards.daily_rate, total_q=total_q,
                    your_q=your_q, est_daily=est_daily, rpc=rpc,
                    spread_cents=ob.spread_cents, cap_needed=cap_needed,
                )

                results.append(ScanResult(
                    market=m,
                    orderbook=ob,
                    total_q_min=total_q,
                    your_q_min=your_q,
                    reward_per_q=reward_per_q,
                    estimated_daily_reward=est_daily,
                    spread_cents=ob.spread_cents,
                    your_depth_share=depth_share,
                    capital_needed=cap_needed,
                    reward_per_capital=rpc,
                ))

                # Rate limit: don't hammer the API
                time.sleep(0.1)

            except Exception as e:
                logger.debug(f"Skip {m.question[:50]}: {e}")
                continue

        # Sort: reward mode by $/capital, normal mode by $/Q
        if self.reward_mode:
            results.sort(key=lambda r: -r.reward_per_capital)
        else:
            results.sort(key=lambda r: -r.reward_per_q)

        elapsed = time.time() - start
        logger.info(
            f"Scan complete: {len(results)} viable markets in {elapsed:.1f}s"
        )

        # Log top 5
        for i, r in enumerate(results[:5]):
            logger.info(
                f"  #{i+1}: ${r.market.rewards.daily_rate}/d | "
                f"Q={r.total_q_min:.0f} | $/Q={r.reward_per_q:.4f} | "
                f"est=${r.estimated_daily_reward:.2f}/d | "
                f"{r.market.question[:50]}"
            )

        return results

    def should_migrate(
        self,
        current_reward_per_q: float,
        best_candidate: ScanResult,
    ) -> bool:
        """Should we migrate from current market to a better one?"""
        if current_reward_per_q <= 0:
            return True
        ratio = best_candidate.reward_per_q / current_reward_per_q
        return ratio >= self.cfg.migrate_threshold
