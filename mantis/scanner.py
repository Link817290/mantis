"""Market scanner - finds optimal markets based on reward/Q ratio and risk factors."""
from __future__ import annotations

import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx

from .config import MantisConfig, EngineConfig
from .polymarket_client import PolymarketClient
from .strategy_log import slog
from .types import Market, Orderbook, ScanResult

logger = logging.getLogger("mantis.scanner")

DATA_API_BASE = "https://data-api.polymarket.com"
MAX_PARALLEL_REQUESTS = 10  # Limit concurrent API calls


@dataclass
class MarketQuality:
    """Quality metrics for market selection."""
    activity_score: float = 0.0      # 0-1, higher = more active
    fill_prob_score: float = 0.0     # 0-1, higher = easier to fill
    volatility_score: float = 0.0    # 0-1, higher = lower volatility (safer)
    vpin_score: float = 0.0          # 0-1, higher = less toxic
    price_score: float = 0.0         # 0-1, higher = better price range
    composite_score: float = 0.0     # weighted average


def fetch_market_trades(http: httpx.Client, condition_id: str, limit: int = 100) -> list[dict]:
    """Fetch recent trades for quality analysis."""
    try:
        resp = http.get(
            f"{DATA_API_BASE}/trades",
            params={"market": condition_id, "limit": str(limit)},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        trades = []
        for t in raw:
            trades.append({
                "price": float(t.get("price", 0)),
                "size": float(t.get("size", 0)),
                "side": t.get("side", "").upper(),
                "timestamp": int(t.get("timestamp", 0)),
            })
        return trades
    except Exception as e:
        logger.debug(f"Failed to fetch trades for {condition_id[:8]}: {e}")
        return []


def calculate_activity_score(trades: list[dict]) -> float:
    """Calculate market activity score based on trade frequency and volume.

    Returns 0-1, higher = more active.
    """
    if len(trades) < 5:
        return 0.0

    # Calculate time span
    timestamps = [t["timestamp"] for t in trades if t["timestamp"] > 0]
    if len(timestamps) < 2:
        return 0.0

    time_span_hours = (max(timestamps) - min(timestamps)) / 3600
    if time_span_hours < 0.1:
        return 0.0

    # Trades per hour
    trades_per_hour = len(trades) / time_span_hours

    # Total volume
    total_volume = sum(t["size"] for t in trades)
    volume_per_hour = total_volume / time_span_hours

    # Score: combine frequency and volume
    # Ideal: 5+ trades/hour, 100+ volume/hour
    freq_score = min(1.0, trades_per_hour / 5.0)
    vol_score = min(1.0, volume_per_hour / 100.0)

    return (freq_score * 0.6 + vol_score * 0.4)


def calculate_volatility_score(trades: list[dict]) -> float:
    """Calculate volatility score based on price movements.

    Returns 0-1, higher = lower volatility (safer).
    """
    if len(trades) < 10:
        return 0.5  # neutral if not enough data

    prices = [t["price"] for t in trades if 0 < t["price"] < 1]
    if len(prices) < 10:
        return 0.5

    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

    if len(returns) < 5:
        return 0.5

    # Standard deviation of returns
    try:
        vol = statistics.stdev(returns)
    except statistics.StatisticsError:
        return 0.5

    # Score: lower volatility = higher score
    # vol < 0.01 = excellent, vol > 0.05 = poor
    if vol < 0.01:
        return 1.0
    elif vol > 0.05:
        return 0.2
    else:
        return 1.0 - (vol - 0.01) / 0.04 * 0.8


def calculate_vpin_score(trades: list[dict]) -> float:
    """Calculate VPIN-based toxicity score.

    Returns 0-1, higher = less toxic.
    """
    if len(trades) < 20:
        return 0.5  # neutral if not enough data

    # Simple VPIN: buy/sell imbalance
    buy_vol = sum(t["size"] for t in trades if t["side"] == "BUY")
    sell_vol = sum(t["size"] for t in trades if t["side"] == "SELL")
    total_vol = buy_vol + sell_vol

    if total_vol < 10:
        return 0.5

    imbalance = abs(buy_vol - sell_vol) / total_vol

    # Score: lower imbalance = higher score (less toxic)
    # imbalance < 0.2 = safe, > 0.5 = toxic
    if imbalance < 0.2:
        return 1.0
    elif imbalance > 0.5:
        return 0.2
    else:
        return 1.0 - (imbalance - 0.2) / 0.3 * 0.8


def calculate_fill_prob_score(orderbook: Orderbook, your_size: float) -> float:
    """Estimate fill probability based on queue depth.

    Returns 0-1, higher = easier to fill.
    """
    if not orderbook.bids or not orderbook.asks:
        return 0.0

    # Queue depth at best prices
    best_bid_depth = sum(l.size for l in orderbook.bids if l.price == orderbook.best_bid)
    best_ask_depth = sum(l.size for l in orderbook.asks if l.price == orderbook.best_ask)

    avg_queue = (best_bid_depth + best_ask_depth) / 2

    # Pro-rata fill probability: your_size / (your_size + queue)
    fill_prob = your_size / (your_size + avg_queue) if avg_queue > 0 else 1.0

    # Also penalize very deep queues
    # Queue < 50 = good, > 200 = bad
    if avg_queue < 50:
        queue_penalty = 1.0
    elif avg_queue > 200:
        queue_penalty = 0.5
    else:
        queue_penalty = 1.0 - (avg_queue - 50) / 150 * 0.5

    return fill_prob * queue_penalty


def calculate_price_score(mid: float) -> float:
    """Calculate price extremity penalty.

    Returns 0-1, higher = safer price range.
    Prices near 0.5 are safest, near 0 or 1 are risky.
    """
    # Distance from 0.5
    dist = abs(mid - 0.5)

    # Score: closer to 0.5 = higher score
    # 0.5 = 1.0, 0.1 or 0.9 = 0.2
    if dist < 0.1:
        return 1.0
    elif dist > 0.4:
        return 0.2
    else:
        return 1.0 - (dist - 0.1) / 0.3 * 0.8


def calculate_composite_quality(quality: MarketQuality) -> float:
    """Calculate weighted composite quality score."""
    weights = {
        "activity": 0.20,      # 20% - need active market
        "fill_prob": 0.20,     # 20% - need to actually fill
        "volatility": 0.25,    # 25% - low vol = less adverse selection
        "vpin": 0.20,          # 20% - avoid toxic flow
        "price": 0.15,         # 15% - avoid extreme prices
    }

    composite = (
        quality.activity_score * weights["activity"] +
        quality.fill_prob_score * weights["fill_prob"] +
        quality.volatility_score * weights["volatility"] +
        quality.vpin_score * weights["vpin"] +
        quality.price_score * weights["price"]
    )

    return composite


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
        self._http = httpx.Client(timeout=15)  # For trade fetching
        self._last_full_scan: float = 0.0
        self._cached_candidates: list[Market] = []  # Cache phase 1 results

    def close(self):
        """Close HTTP client."""
        self._http.close()

    def _fetch_market_data(self, market: Market, force_cids: set[str]) -> dict | None:
        """Fetch orderbook and trades for a single market. Returns data dict or None."""
        try:
            token = market.yes_token
            if not token:
                return None

            is_position_market = market.condition_id in force_cids
            ob = self.client.fetch_orderbook(token.token_id)

            if not ob.bids or not ob.asks:
                if not is_position_market:
                    return None

            # Skip markets with extreme spread (illiquid or stale)
            if ob.spread_cents > 5.0 and not is_position_market:
                return None

            # Skip markets with thin books
            bid_depth_5 = sum(l.size for l in ob.bids[:5])
            ask_depth_5 = sum(l.size for l in ob.asks[:5])
            if (bid_depth_5 < 50 or ask_depth_5 < 50) and not is_position_market:
                return None

            # Fetch trades for quality analysis
            trades = fetch_market_trades(self._http, market.condition_id, limit=100)

            return {
                "market": market,
                "orderbook": ob,
                "trades": trades,
                "is_position_market": is_position_market,
            }
        except Exception as e:
            logger.debug(f"Fetch failed for {market.question[:30]}: {e}")
            return None

    def _parallel_fetch(self, markets: list[Market], force_cids: set[str]) -> list[dict]:
        """Fetch market data in parallel using ThreadPoolExecutor."""
        results = []

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
            futures = {
                executor.submit(self._fetch_market_data, m, force_cids): m
                for m in markets
            }

            for future in as_completed(futures):
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.debug(f"Parallel fetch error: {e}")

        return results

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

        # Phase 2: fetch orderbooks and trades in PARALLEL
        logger.info(f"Phase 2: fetching data for {len(candidates)} candidates in parallel...")
        fetch_start = time.time()
        fetched_data = self._parallel_fetch(candidates, force_cids)
        logger.info(f"Parallel fetch complete: {len(fetched_data)} markets in {time.time() - fetch_start:.1f}s")

        # Phase 3: process fetched data (fast, no I/O)
        results: list[ScanResult] = []
        for data in fetched_data:
            try:
                m = data["market"]
                ob = data["orderbook"]
                trades = data["trades"]
                is_position_market = data["is_position_market"]

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

                # === Quality Scoring ===
                activity_score = calculate_activity_score(trades)
                volatility_score = calculate_volatility_score(trades)
                vpin_score = calculate_vpin_score(trades)
                fill_prob_score = calculate_fill_prob_score(ob, m.rewards.min_size)
                price_score = calculate_price_score(ob.midpoint)

                # Composite quality
                quality = MarketQuality(
                    activity_score=activity_score,
                    fill_prob_score=fill_prob_score,
                    volatility_score=volatility_score,
                    vpin_score=vpin_score,
                    price_score=price_score,
                )
                quality.composite_score = calculate_composite_quality(quality)

                # Risk-adjusted reward = reward * quality
                risk_adjusted = rpc * quality.composite_score

                # Quality filter: reject low-quality markets (but keep position markets)
                if not is_position_market:
                    if quality.composite_score < self.cfg.min_quality_score:
                        logger.debug(
                            f"Skip {m.question[:30]}: low quality {quality.composite_score:.2f}"
                        )
                        continue
                    if activity_score < self.cfg.min_activity_score:
                        logger.debug(
                            f"Skip {m.question[:30]}: low activity {activity_score:.2f}"
                        )
                        continue

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
                    activity_score=activity_score,
                    fill_prob_score=fill_prob_score,
                    volatility_score=volatility_score,
                    vpin_score=vpin_score,
                    price_score=price_score,
                    quality_score=quality.composite_score,
                    risk_adjusted_reward=risk_adjusted,
                ))

            except Exception as e:
                logger.debug(f"Process error for {data.get('market', {})}: {e}")
                continue

        # Sort by RISK-ADJUSTED reward (reward * quality)
        if self.reward_mode:
            results.sort(key=lambda r: -r.risk_adjusted_reward)
        else:
            results.sort(key=lambda r: -r.reward_per_q * r.quality_score)

        elapsed = time.time() - start
        logger.info(
            f"Scan complete: {len(results)} viable markets in {elapsed:.1f}s"
        )

        # Log top 5 with quality breakdown
        for i, r in enumerate(results[:5]):
            logger.info(
                f"  #{i+1}: ${r.market.rewards.daily_rate}/d | "
                f"Q={r.total_q_min:.0f} | rpc=${r.reward_per_capital:.4f} | "
                f"quality={r.quality_score:.2f} | "
                f"adj=${r.risk_adjusted_reward:.4f} | "
                f"{r.market.question[:40]}"
            )
            logger.debug(
                f"       Activity={r.activity_score:.2f} Fill={r.fill_prob_score:.2f} "
                f"Vol={r.volatility_score:.2f} VPIN={r.vpin_score:.2f} Price={r.price_score:.2f}"
            )

        return results

    def quick_scan(self, current_market_ids: set[str]) -> list[tuple[str, float]]:
        """Quick scan using midpoint API only - for frequent opportunity checks.

        Returns list of (condition_id, estimated_rpc) for markets that might be better.
        Does NOT fetch full orderbook or trades - just a fast signal.
        """
        logger.debug("Quick scan starting...")
        start = time.time()

        # Use cached candidates if recent
        if time.time() - self._last_full_scan < 300 and self._cached_candidates:
            candidates = self._cached_candidates
        else:
            all_markets = self.client.fetch_all_sampling_markets()
            candidates = [
                m for m in all_markets
                if m.active and m.yes_token
                and m.rewards.daily_rate >= self.cfg.min_reward_rate
                and m.rewards.min_size <= self.cfg.max_min_size
                and m.condition_id not in current_market_ids
            ]
            candidates.sort(key=lambda m: -m.rewards.daily_rate)
            candidates = candidates[:30]  # Only check top 30

        opportunities = []
        for m in candidates[:20]:  # Quick check top 20
            try:
                token = m.yes_token
                if not token:
                    continue

                # Use midpoint API (faster than full orderbook)
                mid = self.client.fetch_midpoint(token.token_id)
                if mid is None or mid < 0.15 or mid > 0.85:
                    continue

                # Quick spread check
                spread_data = self.client.fetch_spread(token.token_id)
                if spread_data and spread_data["spread"] > 5.0:
                    continue

                # Rough RPC estimate (without full Q calculation)
                cap_needed = m.rewards.min_size * mid * 2
                est_rpc = m.rewards.daily_rate / cap_needed if cap_needed > 0 else 0

                if est_rpc > 0.01:  # Only signal if RPC > 1%
                    opportunities.append((m.condition_id, est_rpc))

            except Exception:
                continue

        logger.debug(f"Quick scan: {len(opportunities)} opportunities in {time.time() - start:.1f}s")
        return opportunities

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
