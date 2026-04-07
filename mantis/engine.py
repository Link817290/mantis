"""Order engine - core market making logic with game theory pricing."""
from __future__ import annotations

import logging
import statistics
import time

import httpx

from .config import MantisConfig
from .db import Database
from .game_theory import (
    CFREngine,
    detect_ecosystem_state,
    glosten_milgrom_min_spread,
    markov_price_drift,
    nash_optimal_spread,
    VPINCalculator,
    EatDetector,
    Randomizer,
    UnwindOptimizer,
)
from .polymarket_client import PolymarketClient, PolymarketTrader
from .strategy_log import slog
from .types import ActiveOrder, MarketState, Orderbook, Position, Side

logger = logging.getLogger("mantis.engine")

DATA_API_BASE = "https://data-api.polymarket.com"


class QualityMonitor:
    """Real-time market quality monitoring."""

    def __init__(self):
        self._http = httpx.Client(timeout=10)
        self._last_fetch: dict[str, float] = {}  # condition_id -> timestamp
        self._fetch_interval = 60  # Fetch trades every 60 seconds

    def close(self):
        self._http.close()

    def update_quality(self, state: MarketState) -> dict:
        """Update quality metrics for a market state.

        Returns dict with: should_widen, should_migrate, alerts
        """
        cid = state.market.condition_id
        now = time.time()
        result = {"should_widen": False, "should_migrate": False, "alerts": []}

        # Only fetch new trades periodically to avoid rate limits
        last_fetch = self._last_fetch.get(cid, 0)
        if now - last_fetch >= self._fetch_interval:
            trades = self._fetch_recent_trades(cid)
            if trades:
                state.recent_trades = trades
                self._last_fetch[cid] = now

        # Calculate quality scores from recent trades
        trades = state.recent_trades
        if len(trades) >= 10:
            # Activity score
            old_activity = state.activity_score
            state.activity_score = self._calc_activity(trades)

            # VPIN score
            old_vpin = state.vpin_score
            state.vpin_score = self._calc_vpin_score(trades)

            # Volatility score
            old_vol = state.volatility_score
            state.volatility_score = self._calc_volatility_score(trades)

            # Composite quality
            old_quality = state.quality_score
            state.quality_score = (
                state.activity_score * 0.2 +
                state.vpin_score * 0.35 +
                state.volatility_score * 0.30 +
                0.15  # Base score for price (already filtered at scan)
            )

            # Track quality trend
            if old_quality > 0:
                state.quality_trend = state.quality_score - old_quality

            state.last_quality_update = now

            # Check for alerts
            if state.vpin_score < 0.3:
                result["alerts"].append(f"HIGH VPIN - toxic flow detected")
                result["should_widen"] = True

            if state.volatility_score < 0.3:
                result["alerts"].append(f"HIGH VOLATILITY - dangerous conditions")
                result["should_widen"] = True

            if state.activity_score < 0.15:
                result["alerts"].append(f"LOW ACTIVITY - market dying")
                result["should_migrate"] = True

            if state.quality_score < 0.25:
                result["alerts"].append(f"QUALITY CRITICAL - consider migration")
                result["should_migrate"] = True

            # Rapid quality deterioration
            if state.quality_trend < -0.15:
                result["alerts"].append(f"QUALITY DROPPING FAST")
                result["should_widen"] = True

            # Log trade flow for analysis (every update)
            buy_vol = sum(t["size"] for t in trades if t["side"] == "BUY")
            sell_vol = sum(t["size"] for t in trades if t["side"] == "SELL")
            total_vol = buy_vol + sell_vol
            net_flow = buy_vol - sell_vol
            vpin_raw = abs(net_flow) / total_vol if total_vol > 0 else 0
            avg_size = total_vol / len(trades) if trades else 0
            large_trades = [t for t in trades if t["size"] > avg_size * 2]
            large_pct = len(large_trades) / len(trades) if trades else 0

            slog.trade_flow(
                market=state.market.question[:40] if hasattr(state, 'market') else cid[:40],
                window_sec=int(now - (trades[-1].get("ts", now) if trades else now)),
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                net_flow=net_flow,
                vpin=vpin_raw,
                n_trades=len(trades),
                avg_trade_size=avg_size,
                large_trade_pct=large_pct,
            )

        return result

    def _fetch_recent_trades(self, condition_id: str) -> list[dict]:
        """Fetch recent trades from API."""
        try:
            resp = self._http.get(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": "100"},
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
            logger.debug(f"Failed to fetch trades: {e}")
            return []

    def _calc_activity(self, trades: list[dict]) -> float:
        """Calculate activity score 0-1."""
        if len(trades) < 5:
            return 0.0
        timestamps = [t["timestamp"] for t in trades if t["timestamp"] > 0]
        if len(timestamps) < 2:
            return 0.0
        hours = (max(timestamps) - min(timestamps)) / 3600
        if hours < 0.1:
            return 0.5  # Very recent trades, assume active
        trades_per_hour = len(trades) / hours
        return min(1.0, trades_per_hour / 5.0)

    def _calc_vpin_score(self, trades: list[dict]) -> float:
        """Calculate VPIN score 0-1 (higher = less toxic)."""
        buy_vol = sum(t["size"] for t in trades if t["side"] == "BUY")
        sell_vol = sum(t["size"] for t in trades if t["side"] == "SELL")
        total = buy_vol + sell_vol
        if total < 10:
            return 0.5
        imbalance = abs(buy_vol - sell_vol) / total
        # imbalance < 0.2 = safe (score 1.0), > 0.5 = toxic (score 0.2)
        if imbalance < 0.2:
            return 1.0
        elif imbalance > 0.5:
            return 0.2
        else:
            return 1.0 - (imbalance - 0.2) / 0.3 * 0.8

    def _calc_volatility_score(self, trades: list[dict]) -> float:
        """Calculate volatility score 0-1 (higher = lower vol)."""
        prices = [t["price"] for t in trades if 0 < t["price"] < 1]
        if len(prices) < 10:
            return 0.5
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = abs(prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        if len(returns) < 5:
            return 0.5
        try:
            vol = statistics.stdev(returns)
        except:
            return 0.5
        # vol < 0.01 = safe (1.0), > 0.05 = dangerous (0.2)
        if vol < 0.01:
            return 1.0
        elif vol > 0.05:
            return 0.2
        else:
            return 1.0 - (vol - 0.01) / 0.04 * 0.8


class OrderEngine:
    def __init__(
        self,
        client: PolymarketClient,
        trader: PolymarketTrader | None,
        db: Database,
        config: MantisConfig,
    ):
        self.client = client
        self.trader = trader
        self.db = db
        self.cfg = config.engine
        self.risk_cfg = config.risk

        # CFR engine
        self.cfr = CFREngine(config.cfr.strategies)
        cfr_state = db.get_cfr_state()
        if cfr_state:
            self.cfr.load_state(cfr_state)
            logger.info(f"Loaded CFR state: {self.cfr.rounds} rounds")

        # 防吃单组件
        self.vpin_calculators: dict[str, VPINCalculator] = {}  # condition_id -> calculator
        self.eat_detectors: dict[str, EatDetector] = {}

        # 实时质量监控
        self.quality_monitor = QualityMonitor()
        self.randomizer = Randomizer()
        self.unwind_optimizer = UnwindOptimizer(
            min_profit_pct=getattr(config.engine, 'unwind_min_profit_pct', 0.005),
            max_loss_pct=getattr(config.engine, 'unwind_max_loss_pct', 0.03),
            time_decay_hours=getattr(config.engine, 'unwind_time_decay_hours', 4.0),
            urgent_loss_pct=getattr(config.engine, 'unwind_urgent_loss_pct', 0.05),
        )

        # 持仓入场时间追踪
        self.position_entry_times: dict[str, float] = {}  # token_id -> timestamp

        # 滑点追踪
        self.slippage_stats: dict[str, list] = {}  # condition_id -> list of slippages
        self.last_orderbook_time: dict[str, float] = {}  # condition_id -> timestamp

        # 订单生命周期追踪
        self.order_place_times: dict[str, float] = {}  # order_id -> place timestamp
        self.order_details: dict[str, dict] = {}  # order_id -> {market, side, price, size}

        # 成交后价格追踪 (用于adverse selection分析)
        self.pending_trajectory_checks: list[dict] = []  # {fill_id, fill_time, market, ...}

        # 持仓同步：从 API 获取真实持仓
        self._proxy_address = config.wallet.browser_address or ""
        self._last_position_sync: float = 0.0
        self._position_sync_interval: float = 30.0  # 每30秒同步一次

    def tick(self, state: MarketState) -> MarketState:
        """Main loop tick for a single market. Called every order_refresh_sec.

        1. Fetch latest orderbook
        2. Record price history
        3. Compute quotes using game theory stack
        4. Compare with existing orders
        5. Reprice if needed
        6. [NEW] Check for eat signals and retreat if needed
        """
        market = state.market
        token = market.yes_token
        if not token:
            return state

        # 初始化防吃单组件
        cid = market.condition_id
        if cid not in self.vpin_calculators:
            self.vpin_calculators[cid] = VPINCalculator()
        if cid not in self.eat_detectors:
            self.eat_detectors[cid] = EatDetector()

        # 1. Fetch orderbook
        try:
            ob = self.client.fetch_orderbook(token.token_id)
            self.last_orderbook_time[cid] = time.time()  # Track fetch time
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return state
        state.orderbook = ob

        if not ob.bids or not ob.asks:
            logger.warning(f"Empty orderbook for {market.question[:40]}")
            return state

        # 更新 EatDetector
        bid_depth = sum(l.size for l in ob.bids[:5])
        ask_depth = sum(l.size for l in ob.asks[:5])
        self.eat_detectors[cid].update(ob.midpoint, bid_depth, ask_depth)

        # [NEW] 实时质量监控
        quality_result = self.quality_monitor.update_quality(state)
        if quality_result["alerts"]:
            for alert in quality_result["alerts"]:
                logger.warning(f"[QUALITY] {market.question[:30]}: {alert}")
                slog.quality_alert(
                    market=market.question[:40],
                    alert=alert,
                    quality=state.quality_score,
                    vpin=state.vpin_score,
                    activity=state.activity_score,
                )

        # 根据质量调整价差
        if quality_result["should_widen"]:
            state.spread_multiplier = max(state.spread_multiplier, 1.5)
            logger.info(f"Quality-based spread widening to {state.spread_multiplier}x")

        # 标记需要迁移
        if quality_result["should_migrate"] and not state.unwind_only:
            logger.warning(f"Quality suggests migration from {market.question[:30]}")
            state.marked_for_exit = True

        # 1a. On first tick, cancel stale exchange orders from previous runs
        if not state.orders_cleaned_up:
            self._cleanup_stale_orders(state)
            state.orders_cleaned_up = True

        # 1b. Sync fills and update positions
        self._sync_fills(state)

        # 1b2. Periodic position sync from API (every 30s)
        self.sync_positions_from_api(state)

        # 1c. [NEW] Check for eat signals - retreat if needed
        self._check_and_retreat(state)

        # 1d. Check take-profit/stop-loss on existing inventory
        self._maybe_take_profit(state)

        # 2. Record price history and compute volatility
        mid = ob.midpoint
        state.recent_prices.append(mid)
        if len(state.recent_prices) > self.cfg.markov_window * 2:
            state.recent_prices = state.recent_prices[-self.cfg.markov_window * 2:]
        self.db.record_price(market.condition_id, mid, ob.spread_cents)

        # Compute rolling volatility from recent prices (proxy for 3h vol)
        if len(state.recent_prices) >= 5:
            returns = [abs(state.recent_prices[i] - state.recent_prices[i-1])
                       for i in range(1, len(state.recent_prices))]
            state.volatility_3h = sum(returns) / len(returns)

        # Log tick state
        yes_inv = sum(p.size for p in state.positions if p.outcome == "Yes")
        slog.tick(
            market=market.question[:40],
            mid=mid, spread_cents=ob.spread_cents,
            vol_3h=state.volatility_3h,
            bid_live=state.bid_cooldown_until <= time.time(),
            ask_live=state.ask_cooldown_until <= time.time(),
            yes_inventory=yes_inv,
            spread_multiplier=state.spread_multiplier,
        )

        # Log competition snapshot (every 5 minutes to reduce volume)
        if int(time.time()) % 300 < self.cfg.order_refresh_sec:
            total_bid_depth = sum(l.size for l in ob.bids)
            total_ask_depth = sum(l.size for l in ob.asks)
            # Estimate number of makers from distinct price levels
            n_bid_levels = len(set(l.price for l in ob.bids))
            n_ask_levels = len(set(l.price for l in ob.asks))
            n_makers_est = max(n_bid_levels, n_ask_levels)
            # Your Q share
            your_q_share = 0.0
            if state.total_q_min > 0 and state.active_orders:
                from .scanner import compute_q_score
                max_spread = market.rewards.max_spread
                your_q = sum(
                    compute_q_score(o.size, abs(o.price - mid) * 100, max_spread)
                    for o in state.active_orders
                )
                your_q_share = your_q / (state.total_q_min + your_q) if your_q > 0 else 0

            slog.competition_snapshot(
                market=market.question[:40],
                n_makers=n_makers_est,
                tightest_spread=ob.spread_cents,
                avg_spread=ob.spread_cents,  # Simplified
                total_bid_depth=total_bid_depth,
                total_ask_depth=total_ask_depth,
                your_q_share=your_q_share,
            )

        # 3. Compute quotes
        bid_price, ask_price, half_spread = self._compute_quotes(state)

        # 4. Check if reprice needed
        needs_reprice = self._needs_reprice(state, bid_price, ask_price)

        if not needs_reprice:
            self._accrue_reward(state)
            return state

        # 5. Cancel old orders and place new ones
        old_bid = None
        old_ask = None
        for o in state.active_orders:
            if o.side == Side.BUY:
                old_bid = o.price
            elif o.side == Side.SELL:
                old_ask = o.price

        slog.reprice(
            market=market.question[:40],
            old_bid=old_bid or 0, old_ask=old_ask or 0,
            new_bid=bid_price, new_ask=ask_price, mid=mid,
        )

        logger.info(
            f"[{market.question[:30]}] Repricing: "
            f"bid={bid_price:.3f} ask={ask_price:.3f} half={half_spread*100:.1f}c"
        )

        self._cancel_existing_orders(state)
        self._place_orders(state, bid_price, ask_price)
        state.last_reprice = time.time()

        # Accrue estimated reward income if we have live orders
        self._accrue_reward(state)

        return state

    def _compute_quotes(
        self, state: MarketState,
    ) -> tuple[float, float, float]:
        """Compute bid/ask prices using the game theory stack.

        改进：
        1. 集成 VPIN 检测订单流毒性
        2. 添加随机抖动防止被预测
        3. VPIN 高时自动加宽价差
        """
        ob = state.orderbook
        assert ob is not None
        mid = ob.midpoint
        market = state.market
        cid = market.condition_id

        # 获取 VPIN
        vpin_calc = self.vpin_calculators.get(cid)
        vpin = vpin_calc.get_vpin() if vpin_calc else 0.15

        # Reward farming mode: skip game theory, place tight
        if self.cfg.reward_mode:
            bid_price, ask_price, half = self._compute_reward_quotes(state)
            # 即使在 reward mode，VPIN 高时也要加宽
            if vpin > 0.4:
                extra_spread = (vpin - 0.4) * 0.02  # VPIN 每高 0.1，加宽 0.2 分
                bid_price = round(bid_price - extra_spread, 3)
                ask_price = round(ask_price + extra_spread, 3)
                logger.info(f"VPIN={vpin:.2f} high, widening reward spread by {extra_spread*100:.1f}c")
            return bid_price, ask_price, half

        # Layer 1: Markov price drift
        drift = markov_price_drift(state.recent_prices)
        drift = max(-0.01, min(0.01, drift))  # cap at 1 cent

        # Layer 2: Glosten-Milgrom minimum spread (使用 VPIN)
        gm_min = glosten_milgrom_min_spread(
            state.recent_fills,
            informed_ratio=0.15,
            expected_jump=0.05,
            vpin=vpin,
        )

        # Layer 3: Nash optimal spread (降低 aggression，更保守)
        aggression = 0.9 if vpin < 0.3 else 0.8 if vpin < 0.5 else 0.7
        nash_spread = nash_optimal_spread(ob, gm_min, aggression=aggression)

        # CFR baseline (if enough rounds)
        if self.cfr.rounds >= 3:
            cfr_half = self.cfr.select_spread()
        else:
            cfr_half = self.cfg.default_half_spread

        # Final half-spread: max of all constraints
        half = max(
            gm_min / 2,
            nash_spread / 2,
            cfr_half,
            self.cfg.min_half_spread,
        )

        # VPIN 驱动的额外加宽
        if vpin > 0.35:
            vpin_multiplier = 1 + (vpin - 0.35) * 2  # VPIN 0.5 时加宽 30%
            half *= vpin_multiplier
            logger.info(f"VPIN={vpin:.2f}, spread multiplier={vpin_multiplier:.2f}")

        # Apply risk-driven spread widening
        half *= state.spread_multiplier

        # Layer 4: Inventory skew adjustment
        bid_adj, ask_adj = self._inventory_adjustment(state)

        # Compute final prices
        bid_price = round(mid + drift - half + bid_adj, 3)
        ask_price = round(mid + drift + half + ask_adj, 3)

        # 添加随机抖动
        bid_price = self.randomizer.jitter_price(bid_price, max_jitter_cents=0.15)
        ask_price = self.randomizer.jitter_price(ask_price, max_jitter_cents=0.15)

        # Sanity: bid must be positive, ask must be < 1
        bid_price = max(0.001, bid_price)
        ask_price = min(0.999, ask_price)

        # Bid must be < ask
        if bid_price >= ask_price:
            bid_price = round(mid - self.cfg.min_half_spread, 3)
            ask_price = round(mid + self.cfg.min_half_spread, 3)

        # CRITICAL: Avoid crossed orders (would be immediately filled as taker)
        # Bid should be < best_ask, Ask should be > best_bid
        if ob.best_ask > 0 and bid_price >= ob.best_ask:
            bid_price = round(ob.best_ask - 0.001, 3)
            logger.warning(f"Bid would cross ask, adjusted to {bid_price}")
        if ob.best_bid > 0 and ask_price <= ob.best_bid:
            ask_price = round(ob.best_bid + 0.001, 3)
            logger.warning(f"Ask would cross bid, adjusted to {ask_price}")

        # Log spread decision for model training
        market_spread = ob.spread_cents
        final_spread = (ask_price - bid_price) * 100
        max_spread_c = state.market.rewards.max_spread
        expected_q = compute_q_score(
            state.market.rewards.min_size,
            final_spread / 2,  # distance from mid
            max_spread_c
        ) if max_spread_c > 0 else 0

        slog.spread_decision(
            market=state.market.question[:40],
            chosen_spread=cfr_half * 200,  # half spread to full spread in cents
            min_spread=self.cfg.min_half_spread * 200,
            vol_adjusted=gm_min,
            inventory_adjusted=(bid_adj - ask_adj) * 100,  # skew in cents
            vpin_adjusted=vpin,
            final_spread=final_spread,
            spread_vs_market=final_spread - market_spread,
            expected_q_score=expected_q,
        )

        return bid_price, ask_price, half

    def _compute_reward_quotes(
        self, state: MarketState,
    ) -> tuple[float, float, float]:
        """Reward farming mode (defiance_cr): distance = 15% of max_spread.

        Q coefficient at 15% = ((1 - 0.15) / 1)^2 = 0.72 — good Q with safety margin.
        Avoids the 0.1c trap where any price move causes adverse fills.
        """
        ob = state.orderbook
        assert ob is not None
        mid = ob.midpoint
        max_spread = state.market.rewards.max_spread

        # Defiance strategy: 15% of max_spread
        dist_c = max_spread * self.cfg.reward_spread_pct
        half = dist_c / 100  # convert cents to price units

        # Floor: never closer than 0.1c
        half = max(half, 0.001)

        # Ceiling: must stay inside max_spread
        if half * 100 >= max_spread:
            half = (max_spread - 0.1) / 100

        # Book imbalance skew: shift away from heavy side
        bid_skew, ask_skew = self._book_imbalance_skew(ob, half)

        bid_price = round(mid - half + bid_skew, 3)
        ask_price = round(mid + half + ask_skew, 3)

        # Sanity
        bid_price = max(0.001, bid_price)
        ask_price = min(0.999, ask_price)

        if bid_price >= ask_price:
            bid_price = round(mid - half, 3)
            ask_price = round(mid + half, 3)

        # CRITICAL: Avoid crossed orders (would be immediately filled as taker)
        if ob.best_ask > 0 and bid_price >= ob.best_ask:
            bid_price = round(ob.best_ask - 0.001, 3)
            logger.warning(f"[Reward] Bid would cross ask, adjusted to {bid_price}")
        if ob.best_bid > 0 and ask_price <= ob.best_bid:
            ask_price = round(ob.best_bid + 0.001, 3)
            logger.warning(f"[Reward] Ask would cross bid, adjusted to {ask_price}")

        return bid_price, ask_price, half

    def _book_imbalance_skew(
        self, ob: Orderbook, half: float,
    ) -> tuple[float, float]:
        """Detect orderbook imbalance and skew quotes away from the heavy side.

        If bids are much heavier than asks, price likely to drop → shift our
        quotes down slightly to avoid getting filled on bid side.
        Returns (bid_skew, ask_skew) in price units.
        """
        bid_depth = sum(l.size for l in ob.bids[:5])
        ask_depth = sum(l.size for l in ob.asks[:5])
        total = bid_depth + ask_depth
        if total < 1:
            return 0.0, 0.0

        # imbalance: positive = bid-heavy (price likely to drop)
        imbalance = (bid_depth - ask_depth) / total  # [-1, 1]

        # Max skew = 30% of half-spread
        skew = imbalance * half * 0.3
        return -skew, -skew  # shift both prices in same direction

    def _inventory_adjustment(self, state: MarketState) -> tuple[float, float]:
        """Adjust prices based on inventory imbalance.

        If holding too much YES → lower bid (discourage buying more YES)
        If holding too much NO → raise ask
        """
        positions = state.positions
        if not positions:
            return 0.0, 0.0

        yes_value = 0.0
        no_value = 0.0
        for pos in positions:
            if pos.outcome == "Yes":
                yes_value = pos.size * (state.orderbook.midpoint if state.orderbook else 0.5)
            elif pos.outcome == "No":
                no_value = pos.size * (1 - (state.orderbook.midpoint if state.orderbook else 0.5))

        total = yes_value + no_value
        if total < 1.0:
            return 0.0, 0.0

        # Skew ratio: positive = too much YES, negative = too much NO
        skew = (yes_value - no_value) / total

        # Adjustment: shift prices to attract offsetting flow
        # Max adjustment = 1 cent
        adj = skew * 0.01

        bid_adj = -adj  # If YES-heavy, lower bid
        ask_adj = -adj  # If YES-heavy, lower ask too (keep spread constant, shift down)

        return bid_adj, ask_adj

    def _needs_reprice(
        self, state: MarketState, new_bid: float, new_ask: float,
    ) -> bool:
        """Check if we need to update orders."""
        if not state.active_orders:
            return True

        # Find current bid/ask orders
        current_bid = None
        current_ask = None
        for order in state.active_orders:
            if order.side == Side.BUY:
                current_bid = order.price
            elif order.side == Side.SELL:
                current_ask = order.price

        if current_bid is None or current_ask is None:
            return True

        # Reprice if change exceeds threshold
        threshold = self.cfg.reprice_threshold
        if abs(new_bid - current_bid) >= threshold:
            return True
        if abs(new_ask - current_ask) >= threshold:
            return True

        return False

    def _check_and_retreat(self, state: MarketState):
        """检查是否有被吃单的风险，如果有则撤单。

        这是防吃单的核心逻辑：
        1. 检测价格动量
        2. ���测深度消耗
        3. 如果有危险信号，立即撤单
        """
        if not state.active_orders or not state.orderbook:
            return

        cid = state.market.condition_id
        detector = self.eat_detectors.get(cid)
        if not detector:
            return

        mid = state.orderbook.midpoint
        orders_to_cancel = []

        for order in state.active_orders:
            if order.side == Side.BUY:
                should_retreat, reason = detector.should_retreat("bid", order.price, mid)
            else:
                should_retreat, reason = detector.should_retreat("ask", order.price, mid)

            if should_retreat:
                orders_to_cancel.append((order, reason))

        # 撤单
        for order, reason in orders_to_cancel:
            logger.warning(
                f"[RETREAT] Canceling {order.side.value} @{order.price} - {reason}"
            )
            slog.retreat(
                market=state.market.question[:40],
                side=order.side.value,
                price=order.price,
                reason=reason,
            )
            if self.trader:
                try:
                    self.trader.cancel_order(order.order_id)
                    self.db.mark_order_cancelled(order.order_id)
                except Exception as e:
                    logger.error(f"Failed to cancel order in retreat: {e}")

            state.active_orders.remove(order)

        # 如果撤单了，增加冷却时间
        if orders_to_cancel:
            cooldown = self.cfg.post_fill_cooldown_sec * 0.5  # 撤单冷却比成交冷却短
            now = time.time()
            for order, _ in orders_to_cancel:
                if order.side == Side.BUY:
                    state.bid_cooldown_until = max(state.bid_cooldown_until, now + cooldown)
                else:
                    state.ask_cooldown_until = max(state.ask_cooldown_until, now + cooldown)

    def _cleanup_stale_orders(self, state: MarketState):
        """Cancel stale exchange orders from previous runs on first tick.

        After restart, state.active_orders is empty but the exchange may
        still have live orders that consume token balance.
        """
        if not self.trader:
            return

        market = state.market
        token_ids = set()
        if market.yes_token:
            token_ids.add(market.yes_token.token_id)
        if market.no_token:
            token_ids.add(market.no_token.token_id)

        if not token_ids:
            return

        try:
            open_orders = self.trader.get_open_orders()
            stale = [o for o in open_orders
                     if o.get("asset_id") in token_ids or o.get("token_id") in token_ids]
            if stale:
                logger.info(
                    f"[{market.question[:30]}] Cleaning up {len(stale)} stale orders from exchange"
                )
                for o in stale:
                    oid = o.get("id") or o.get("orderID") or o.get("order_id", "")
                    try:
                        self.trader.cancel_order(oid)
                        logger.info(f"  Cancelled stale order {oid[:16]}...")
                    except Exception as e:
                        logger.warning(f"  Failed to cancel stale {oid[:16]}: {e}")
        except Exception as e:
            logger.error(f"Failed to fetch open orders for cleanup: {e}")

    def _cancel_existing_orders(self, state: MarketState):
        """Cancel all existing orders for this market."""
        if not self.trader:
            logger.info("[DRY RUN] Would cancel orders")
            for order in state.active_orders:
                slog.api_order_cancel(
                    market=state.market.question[:40],
                    order_id=order.order_id, reason="reprice_dry",
                )
            state.active_orders.clear()
            return

        for order in state.active_orders:
            try:
                self.trader.cancel_order(order.order_id)
                self.db.mark_order_cancelled(order.order_id)
                slog.api_order_cancel(
                    market=state.market.question[:40],
                    order_id=order.order_id, reason="reprice",
                )
            except Exception as e:
                logger.error(f"Failed to cancel {order.order_id}: {e}")
                slog.api_error(
                    method="cancel", endpoint="cancel_order",
                    error=str(e), order_id=order.order_id,
                )

        state.active_orders.clear()

    def _compute_orderbook_metrics(self, ob: Orderbook) -> dict:
        """Compute orderbook metrics for logging and analysis."""
        if not ob or not ob.bids or not ob.asks:
            return {}

        mid = ob.midpoint
        # Depth within 5 cents of mid
        bid_depth_5c = sum(l.size for l in ob.bids if mid - l.price <= 0.05)
        ask_depth_5c = sum(l.size for l in ob.asks if l.price - mid <= 0.05)
        # Depth within 10 cents
        bid_depth_10c = sum(l.size for l in ob.bids if mid - l.price <= 0.10)
        ask_depth_10c = sum(l.size for l in ob.asks if l.price - mid <= 0.10)

        total_near = bid_depth_5c + ask_depth_5c
        imbalance = (bid_depth_5c - ask_depth_5c) / total_near if total_near > 0 else 0

        return {
            "mid": mid,
            "spread_cents": ob.spread_cents,
            "best_bid": ob.best_bid,
            "best_ask": ob.best_ask,
            "bid_depth_5c": bid_depth_5c,
            "ask_depth_5c": ask_depth_5c,
            "bid_depth_10c": bid_depth_10c,
            "ask_depth_10c": ask_depth_10c,
            "imbalance": imbalance,
            "n_bid_levels": len(ob.bids),
            "n_ask_levels": len(ob.asks),
        }

    def _log_orderbook_snapshot(self, state: MarketState):
        """Log orderbook snapshot for model training."""
        ob = state.orderbook
        if not ob:
            return
        metrics = self._compute_orderbook_metrics(ob)
        if metrics:
            slog.orderbook_snapshot(
                market=state.market.question[:40],
                **metrics
            )

    def _track_order_placed(self, order_id: str, market: str, side: str,
                            price: float, size: float, state: MarketState):
        """Track order placement for lifecycle analysis."""
        now = time.time()
        self.order_place_times[order_id] = now
        self.order_details[order_id] = {
            "market": market,
            "side": side,
            "price": price,
            "size": size,
        }

        # Calculate queue position (orders ahead at same price)
        queue_ahead = 0.0
        ob = state.orderbook
        if ob:
            if side == "BUY":
                for level in ob.bids:
                    if level.price == price:
                        queue_ahead = level.size  # Simplified: all size at price is ahead
                        break
            else:
                for level in ob.asks:
                    if level.price == price:
                        queue_ahead = level.size
                        break

        slog.order_lifecycle(
            market=market[:40],
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            event="placed",
            age_sec=0,
            queue_ahead=queue_ahead,
        )

    def _track_order_filled(self, order_id: str, fill_price: float):
        """Track order fill for lifecycle analysis."""
        place_time = self.order_place_times.get(order_id, 0)
        details = self.order_details.get(order_id, {})
        if not details:
            return

        age_sec = time.time() - place_time if place_time > 0 else 0

        slog.order_lifecycle(
            market=details.get("market", "")[:40],
            order_id=order_id,
            side=details.get("side", ""),
            price=fill_price,
            size=details.get("size", 0),
            event="full_fill",
            age_sec=age_sec,
        )

        # Cleanup
        self.order_place_times.pop(order_id, None)
        self.order_details.pop(order_id, None)

    def _track_order_cancelled(self, order_id: str, reason: str = "reprice"):
        """Track order cancellation for lifecycle analysis."""
        place_time = self.order_place_times.get(order_id, 0)
        details = self.order_details.get(order_id, {})
        if not details:
            return

        age_sec = time.time() - place_time if place_time > 0 else 0

        slog.order_lifecycle(
            market=details.get("market", "")[:40],
            order_id=order_id,
            side=details.get("side", ""),
            price=details.get("price", 0),
            size=details.get("size", 0),
            event="cancelled",
            age_sec=age_sec,
        )

        # Cleanup
        self.order_place_times.pop(order_id, None)
        self.order_details.pop(order_id, None)

    def _process_trajectory_checks(self, states: dict[str, MarketState]):
        """Process pending price trajectory checks for adverse selection analysis."""
        if not self.pending_trajectory_checks:
            return

        now = time.time()
        check_intervals = [10, 30, 60, 300]  # seconds after fill
        completed = []

        for i, check in enumerate(self.pending_trajectory_checks):
            fill_time = check["fill_time"]
            elapsed = now - fill_time
            cid = check["condition_id"]

            # Get current mid price from state
            current_mid = None
            if cid in states and states[cid].orderbook:
                current_mid = states[cid].orderbook.midpoint

            # Record prices at each interval
            for interval in check_intervals:
                if interval not in check["checks_done"] and elapsed >= interval:
                    check[f"mid_after_{interval}s"] = current_mid
                    check["checks_done"].append(interval)

            # If all checks done (300s passed), log the trajectory
            if 300 in check["checks_done"]:
                fill_side = check["fill_side"]
                fill_price = check["fill_price"]
                mid_at_fill = check["mid_at_fill"]

                # Calculate max adverse move
                mids = [
                    check.get(f"mid_after_{t}s") for t in check_intervals
                    if check.get(f"mid_after_{t}s") is not None
                ]
                if mids:
                    if fill_side == "BUY":
                        # Adverse for buy = price went down after we bought
                        max_adverse = max(0, mid_at_fill - min(mids))
                    else:
                        # Adverse for sell = price went up after we sold
                        max_adverse = max(0, max(mids) - mid_at_fill)
                else:
                    max_adverse = 0

                # Calculate realized P&L (simplified)
                mid_300s = check.get("mid_after_300s", mid_at_fill)
                if mid_300s:
                    if fill_side == "BUY":
                        realized_pnl = (mid_300s - fill_price) * 100  # in cents
                    else:
                        realized_pnl = (fill_price - mid_300s) * 100
                else:
                    realized_pnl = 0

                slog.price_trajectory(
                    market=check["market"],
                    fill_id=check["fill_id"],
                    fill_side=fill_side,
                    fill_price=fill_price,
                    mid_at_fill=mid_at_fill,
                    mid_after_10s=check.get("mid_after_10s", 0) or 0,
                    mid_after_30s=check.get("mid_after_30s", 0) or 0,
                    mid_after_60s=check.get("mid_after_60s", 0) or 0,
                    mid_after_300s=check.get("mid_after_300s", 0) or 0,
                    max_adverse_move=max_adverse * 100,  # in cents
                    realized_pnl=realized_pnl,
                )

                # Record to DB for model training
                self.db.record_trajectory(
                    fill_id=check["fill_id"],
                    condition_id=cid,
                    fill_side=fill_side,
                    fill_price=fill_price,
                    mid_at_fill=mid_at_fill,
                    mid_after_10s=check.get("mid_after_10s", 0) or 0,
                    mid_after_30s=check.get("mid_after_30s", 0) or 0,
                    mid_after_60s=check.get("mid_after_60s", 0) or 0,
                    mid_after_300s=check.get("mid_after_300s", 0) or 0,
                    max_adverse_move=max_adverse * 100,
                    realized_pnl=realized_pnl,
                )

                completed.append(i)

        # Remove completed checks (in reverse order to preserve indices)
        for i in reversed(completed):
            self.pending_trajectory_checks.pop(i)

        # Also cleanup old checks that never completed (> 10 min old)
        self.pending_trajectory_checks = [
            c for c in self.pending_trajectory_checks
            if now - c["fill_time"] < 600
        ]

    def sync_positions_from_api(self, state: MarketState) -> bool:
        """从 API 直接获取真实持仓，覆盖本地计算值。

        Returns True if positions were updated.
        """
        now = time.time()

        # 限制同步频率
        if now - self._last_position_sync < self._position_sync_interval:
            return False

        if not self._proxy_address:
            # 尝试从 trader 获取地址
            if self.trader:
                try:
                    self.trader._ensure_client()
                    self._proxy_address = self.trader._client.get_address()
                except Exception:
                    pass
            if not self._proxy_address:
                return False

        try:
            # 从 API 获取所有持仓
            all_positions = self.client.fetch_positions(self._proxy_address)
            self._last_position_sync = now

            # 筛选当前市场的持仓
            cid = state.market.condition_id
            market_positions = [p for p in all_positions if p.get("conditionId") == cid]

            # 清空本地持仓，用 API 数据覆盖
            old_positions = state.positions.copy()
            state.positions.clear()

            for p in market_positions:
                size = float(p.get("size", 0))
                if size <= 0:
                    continue

                from .types import Position
                pos = Position(
                    token_id=p.get("asset", ""),
                    outcome=p.get("outcome", "Unknown"),
                    size=size,
                    avg_cost=float(p.get("avgPrice", 0)),
                )
                state.positions.append(pos)

                # 同步到 DB
                self.db.update_position(
                    token_id=pos.token_id,
                    condition_id=cid,
                    outcome=pos.outcome,
                    size=pos.size,
                    avg_cost=pos.avg_cost,
                )

            # 检测持仓变化
            old_yes = sum(p.size for p in old_positions if p.outcome == "Yes")
            old_no = sum(p.size for p in old_positions if p.outcome == "No")
            new_yes = sum(p.size for p in state.positions if p.outcome == "Yes")
            new_no = sum(p.size for p in state.positions if p.outcome == "No")

            if abs(old_yes - new_yes) > 0.01 or abs(old_no - new_no) > 0.01:
                logger.info(
                    f"Position sync: Yes {old_yes:.1f}->{new_yes:.1f}, "
                    f"No {old_no:.1f}->{new_no:.1f}"
                )
                return True

            return False

        except Exception as e:
            logger.debug(f"Position sync failed: {e}")
            return False

    def force_position_sync(self, state: MarketState):
        """强制立即同步持仓（成交后调用）"""
        self._last_position_sync = 0  # 重置计时器
        self.sync_positions_from_api(state)

    def _check_orderbook_fresh(self, state: MarketState) -> bool:
        """Check if orderbook is fresh enough for placing orders."""
        cid = state.market.condition_id
        last_fetch = self.last_orderbook_time.get(cid, 0)
        age = time.time() - last_fetch
        max_age = getattr(self.cfg, 'orderbook_max_age_sec', 5.0)

        if age > max_age:
            logger.warning(f"Orderbook stale ({age:.1f}s > {max_age}s), skipping order placement")
            slog.orderbook_stale(
                market=state.market.question[:40],
                age_sec=age,
                max_age=max_age,
            )
            return False
        return True

    def _apply_slippage_buffer(self, bid_price: float, ask_price: float,
                                best_bid: float, best_ask: float) -> tuple[float, float]:
        """Apply slippage buffer to avoid becoming taker."""
        buffer = getattr(self.cfg, 'slippage_buffer_cents', 0.3) / 100  # cents to price

        # Ensure our bid is safely below best ask
        safe_bid = min(bid_price, best_ask - buffer)
        # Ensure our ask is safely above best bid
        safe_ask = max(ask_price, best_bid + buffer)

        if safe_bid != bid_price:
            logger.debug(f"Slippage buffer: bid {bid_price:.3f} -> {safe_bid:.3f}")
        if safe_ask != ask_price:
            logger.debug(f"Slippage buffer: ask {ask_price:.3f} -> {safe_ask:.3f}")

        return round(safe_bid, 3), round(safe_ask, 3)

    def _verify_prices_before_place(self, state: MarketState,
                                     bid_price: float, ask_price: float) -> tuple[bool, str]:
        """Verify prices are still valid before placing. Returns (ok, reason)."""
        ob = state.orderbook
        if not ob:
            return False, "no_orderbook"

        # Check for crossed orders (would become taker)
        if bid_price >= ob.best_ask:
            return False, f"bid_crosses_ask: {bid_price} >= {ob.best_ask}"
        if ask_price <= ob.best_bid:
            return False, f"ask_crosses_bid: {ask_price} <= {ob.best_bid}"

        # Check for excessive slippage from mid
        mid = ob.midpoint
        max_slip = getattr(self.cfg, 'max_slippage_cents', 1.0) / 100

        bid_slip = mid - bid_price
        ask_slip = ask_price - mid

        if bid_slip > max_slip * 2:  # Allow wider spread but not too much
            logger.warning(f"Bid too far from mid: {bid_slip*100:.1f}c")
        if ask_slip > max_slip * 2:
            logger.warning(f"Ask too far from mid: {ask_slip*100:.1f}c")

        return True, "ok"

    def _place_orders(self, state: MarketState, bid_price: float, ask_price: float):
        """Place new bid and ask orders."""
        market = state.market
        token = market.yes_token
        if not token:
            return

        # Check orderbook freshness
        if not self._check_orderbook_fresh(state):
            return

        # Apply slippage buffer
        if state.orderbook:
            bid_price, ask_price = self._apply_slippage_buffer(
                bid_price, ask_price,
                state.orderbook.best_bid, state.orderbook.best_ask
            )

        # Verify prices are still valid
        ok, reason = self._verify_prices_before_place(state, bid_price, ask_price)
        if not ok:
            logger.warning(f"Price verification failed: {reason}")
            return

        # Log orderbook snapshot for model training
        self._log_orderbook_snapshot(state)

        # Check inventory — sides with inventory don't need USDC
        yes_pos = next((p for p in state.positions if p.outcome == "Yes" and p.size > 0), None)
        no_pos = next((p for p in state.positions if p.outcome == "No" and p.size > 0), None)

        # UNWIND_ONLY MODE: only place orders that close positions
        if state.unwind_only:
            self._place_unwind_orders(state, bid_price, ask_price, yes_pos, no_pos)
            return

        # Calculate actual capital needed for each side
        # Apply safety margin to avoid insufficient funds errors
        safety_margin = getattr(self.cfg, 'capital_safety_margin', 0.05)
        total_capital = max(0, state.allocated_capital * (1 - safety_margin))

        bid_needs_capital = not (no_pos and no_pos.size > 0)   # SELL No doesn't need USDC
        ask_needs_capital = not (yes_pos and yes_pos.size > 0)  # SELL Yes doesn't need USDC

        has_inventory = yes_pos is not None or no_pos is not None

        if total_capital <= 0 and not has_inventory:
            logger.info("No capital and no inventory — skipping order placement")
            return

        min_s = market.rewards.min_size

        # Calculate actual cost for min_size on each side
        # Add slippage buffer to costs
        slippage_buffer = getattr(self.cfg, 'slippage_buffer_cents', 0.3) / 100
        bid_unit_cost = bid_price + slippage_buffer  # Cost to buy 1 Yes share (with buffer)
        ask_unit_cost = (1 - ask_price) + slippage_buffer  # Cost to buy 1 No share (with buffer)

        bid_min_cost = min_s * bid_unit_cost if bid_needs_capital else 0
        ask_min_cost = min_s * ask_unit_cost if ask_needs_capital else 0
        total_min_cost = bid_min_cost + ask_min_cost

        if self.cfg.reward_mode:
            # Reward mode: try to place min_size on both sides
            if total_min_cost <= total_capital:
                # Can afford both sides
                bid_size = min_s if bid_needs_capital or (no_pos and no_pos.size >= min_s) else 0
                ask_size = min_s if ask_needs_capital or (yes_pos and yes_pos.size >= min_s) else 0
                bid_capital = bid_min_cost
                ask_capital = ask_min_cost
            elif not bid_needs_capital and ask_min_cost <= total_capital:
                # Bid uses inventory, can afford ask
                bid_size = min_s if no_pos and no_pos.size >= min_s else 0
                ask_size = min_s
                bid_capital = 0
                ask_capital = ask_min_cost
            elif not ask_needs_capital and bid_min_cost <= total_capital:
                # Ask uses inventory, can afford bid
                bid_size = min_s
                ask_size = min_s if yes_pos and yes_pos.size >= min_s else 0
                bid_capital = bid_min_cost
                ask_capital = 0
            else:
                # Can't afford both, try to afford at least one
                # Prioritize the cheaper side
                if bid_min_cost <= total_capital and bid_min_cost <= ask_min_cost:
                    bid_size = min_s
                    ask_size = 0
                    bid_capital = bid_min_cost
                    ask_capital = 0
                    logger.warning(f"Only enough capital for bid side (need ${total_min_cost:.2f}, have ${total_capital:.2f})")
                elif ask_min_cost <= total_capital:
                    bid_size = 0
                    ask_size = min_s
                    bid_capital = 0
                    ask_capital = ask_min_cost
                    logger.warning(f"Only enough capital for ask side (need ${total_min_cost:.2f}, have ${total_capital:.2f})")
                else:
                    bid_size = 0
                    ask_size = 0
                    bid_capital = 0
                    ask_capital = 0
                    logger.warning(f"Not enough capital for either side (need ${total_min_cost:.2f}, have ${total_capital:.2f})")
        else:
            # Non-reward mode: maximize size while balanced
            # Split capital proportionally to costs
            if total_min_cost > 0 and total_capital > 0:
                bid_ratio = bid_min_cost / total_min_cost if bid_needs_capital else 0
                ask_ratio = ask_min_cost / total_min_cost if ask_needs_capital else 0
                # Normalize if one side uses inventory
                if bid_ratio + ask_ratio > 0:
                    bid_capital = total_capital * bid_ratio / (bid_ratio + ask_ratio) if bid_needs_capital else 0
                    ask_capital = total_capital * ask_ratio / (bid_ratio + ask_ratio) if ask_needs_capital else 0
                else:
                    bid_capital = 0
                    ask_capital = 0
            else:
                bid_capital = 0
                ask_capital = 0

            # Calculate sizes based on allocated capital
            if bid_capital > 0:
                affordable_bid = int(bid_capital / bid_unit_cost)
                bid_size = affordable_bid if affordable_bid >= min_s else 0
            else:
                bid_size = 0
            if ask_capital > 0:
                affordable_ask = int(ask_capital / ask_unit_cost)
                ask_size = affordable_ask if affordable_ask >= min_s else 0
            else:
                ask_size = 0

        # When selling inventory, use position size (only if >= min_size for rewards)
        if bid_size == 0 and no_pos and no_pos.size >= min_s:
            bid_size = min(int(no_pos.size), min_s)  # Use min_s for reward eligibility
        if ask_size == 0 and yes_pos and yes_pos.size >= min_s:
            ask_size = min(int(yes_pos.size), min_s)  # Use min_s for reward eligibility

        # Check inventory limits before placing
        bid_ok, ask_ok = self._check_inventory_limits(state)

        # Post-fill cooldown: don't re-place a side that just got filled
        now = time.time()
        if now < state.bid_cooldown_until:
            bid_ok = False
            logger.info(f"Bid on cooldown for {state.bid_cooldown_until - now:.0f}s more")
        if now < state.ask_cooldown_until:
            ask_ok = False
            logger.info(f"Ask on cooldown for {state.ask_cooldown_until - now:.0f}s more")

        if bid_size == 0:
            bid_ok = False
            logger.info(f"Can't afford min_size {market.rewards.min_size} on bid side")
        if ask_size == 0:
            ask_ok = False
            logger.info(f"Can't afford min_size {market.rewards.min_size} on ask side")

        if not self.trader:
            # Dry run mode
            logger.info(f"[DRY RUN] BUY {bid_size}@{bid_price} | SELL {ask_size}@{ask_price}")
            if bid_ok:
                oid = f"dry-bid-{time.time():.0f}"
                state.active_orders.append(ActiveOrder(
                    order_id=oid,
                    token_id=token.token_id,
                    side=Side.BUY,
                    price=bid_price,
                    size=bid_size,
                    market_condition_id=market.condition_id,
                ))
                slog.api_order_submit(
                    market=market.question[:40], side="BUY",
                    price=bid_price, size=bid_size, order_id=oid, dry_run=True,
                )
            if ask_ok:
                oid = f"dry-ask-{time.time():.0f}"
                state.active_orders.append(ActiveOrder(
                    order_id=oid,
                    token_id=token.token_id,
                    side=Side.SELL,
                    price=ask_price,
                    size=ask_size,
                    market_condition_id=market.condition_id,
                ))
                slog.api_order_submit(
                    market=market.question[:40], side="SELL",
                    price=ask_price, size=ask_size, order_id=oid, dry_run=True,
                )
            return

        # Place real orders
        # Key insight: prefer SELL when holding inventory (recovers USDC)
        #   Bid side: SELL No (if holding) or BUY Yes (costs USDC)
        #   Ask side: SELL Yes (if holding) or BUY No (costs USDC)
        no_token = market.no_token
        yes_pos = next((p for p in state.positions if p.outcome == "Yes" and p.size > 0), None)
        no_pos = next((p for p in state.positions if p.outcome == "No" and p.size > 0), None)

        # Track order placement results for two-sided validation
        bid_placed = False
        ask_placed = False
        bid_order_id = None
        ask_order_id = None

        if bid_ok:
            if no_pos and no_pos.size >= bid_size:
                # SELL No tokens at (1 - bid_price) → equivalent to bidding Yes at bid_price
                # This recovers USDC instead of spending it!
                no_sell_price = round(1 - bid_price, 3)
                try:
                    result = self.trader.create_limit_order(
                        no_pos.token_id, "SELL", no_sell_price, bid_size,
                    )
                    order_id = result.get("orderID", result.get("id", "unknown"))
                    self.db.record_order(
                        order_id, market.condition_id, no_pos.token_id,
                        "BUY", bid_price, bid_size,
                    )
                    state.active_orders.append(ActiveOrder(
                        order_id=order_id,
                        token_id=no_pos.token_id,
                        side=Side.BUY,  # Internal: bid side
                        price=bid_price,
                        size=bid_size,
                        market_condition_id=market.condition_id,
                        is_sell_no=True,  # Flag: this is actually SELL No
                    ))
                    slog.api_order_submit(
                        market=market.question[:40], side="BID(SELL_NO)",
                        price=bid_price, size=bid_size, order_id=order_id,
                    )
                    logger.info(f"Bid via SELL No @{no_sell_price} (recovers USDC)")
                    bid_placed = True
                    bid_order_id = order_id
                except Exception as e:
                    logger.error(f"Failed to place bid (SELL No): {e}")
                    state.bid_cooldown_until = time.time() + 30  # Cooldown on failure
            else:
                # BUY Yes with USDC capital
                if bid_capital > 0 and bid_size > 0:
                    try:
                        result = self.trader.create_limit_order(
                            token.token_id, "BUY", bid_price, bid_size,
                        )
                        order_id = result.get("orderID", result.get("id", "unknown"))
                        self.db.record_order(
                            order_id, market.condition_id, token.token_id,
                            "BUY", bid_price, bid_size,
                        )
                        state.active_orders.append(ActiveOrder(
                            order_id=order_id,
                            token_id=token.token_id,
                            side=Side.BUY,
                            price=bid_price,
                            size=bid_size,
                            market_condition_id=market.condition_id,
                        ))
                        slog.api_order_submit(
                            market=market.question[:40], side="BUY",
                            price=bid_price, size=bid_size, order_id=order_id,
                        )
                        logger.info(f"Bid via BUY Yes @{bid_price}")
                        bid_placed = True
                        bid_order_id = order_id
                    except Exception as e:
                        logger.error(f"Failed to place bid (BUY Yes): {e}")
                        state.bid_cooldown_until = time.time() + 30  # Cooldown on failure

        if ask_ok:
            if yes_pos and yes_pos.size >= ask_size:
                # SELL Yes tokens at ask_price → recovers USDC!
                try:
                    result = self.trader.create_limit_order(
                        yes_pos.token_id, "SELL", ask_price, ask_size,
                    )
                    order_id = result.get("orderID", result.get("id", "unknown"))
                    self.db.record_order(
                        order_id, market.condition_id, yes_pos.token_id,
                        "SELL", ask_price, ask_size,
                    )
                    state.active_orders.append(ActiveOrder(
                        order_id=order_id,
                        token_id=yes_pos.token_id,
                        side=Side.SELL,  # Internal: ask side
                        price=ask_price,
                        size=ask_size,
                        market_condition_id=market.condition_id,
                        is_sell_yes=True,  # Flag: this is actually SELL Yes
                    ))
                    slog.api_order_submit(
                        market=market.question[:40], side="ASK(SELL_YES)",
                        price=ask_price, size=ask_size, order_id=order_id,
                    )
                    logger.info(f"Ask via SELL Yes @{ask_price} (recovers USDC)")
                    ask_placed = True
                    ask_order_id = order_id
                except Exception as e:
                    logger.error(f"Failed to place ask (SELL Yes): {e}")
                    state.ask_cooldown_until = time.time() + 30
            else:
                # BUY No (equivalent to SELL Yes side) with USDC capital
                if ask_capital > 0 and ask_size > 0 and no_token:
                    no_buy_price = round(1 - ask_price, 3)
                    try:
                        result = self.trader.create_limit_order(
                            no_token.token_id, "BUY", no_buy_price, ask_size,
                        )
                        order_id = result.get("orderID", result.get("id", "unknown"))
                        self.db.record_order(
                            order_id, market.condition_id, no_token.token_id,
                            "SELL", ask_price, ask_size,
                        )
                        state.active_orders.append(ActiveOrder(
                            order_id=order_id,
                            token_id=no_token.token_id,
                            side=Side.SELL,
                            price=ask_price,
                            size=ask_size,
                            market_condition_id=market.condition_id,
                        ))
                        slog.api_order_submit(
                            market=market.question[:40], side="ASK(BUY_NO)",
                            price=ask_price, size=ask_size, order_id=order_id,
                        )
                        logger.info(f"Ask via BUY No @{no_buy_price}")
                        ask_placed = True
                        ask_order_id = order_id
                    except Exception as e:
                        logger.error(f"Failed to place ask (BUY No): {e}")
                        state.ask_cooldown_until = time.time() + 30

        # TWO-SIDED VALIDATION: If only one side succeeded, cancel it to avoid imbalance
        if bid_ok and ask_ok:  # Both sides were attempted
            if bid_placed and not ask_placed:
                logger.warning(f"ONE-SIDED: Bid succeeded but ask failed, canceling bid")
                if bid_order_id:
                    try:
                        self.trader.cancel_order(bid_order_id)
                        # Remove from active_orders
                        state.active_orders = [o for o in state.active_orders
                                               if o.order_id != bid_order_id]
                        logger.info(f"Canceled one-sided bid {bid_order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel one-sided bid: {e}")
                state.bid_cooldown_until = time.time() + 60  # Longer cooldown

            elif ask_placed and not bid_placed:
                logger.warning(f"ONE-SIDED: Ask succeeded but bid failed, canceling ask")
                if ask_order_id:
                    try:
                        self.trader.cancel_order(ask_order_id)
                        state.active_orders = [o for o in state.active_orders
                                               if o.order_id != ask_order_id]
                        logger.info(f"Canceled one-sided ask {ask_order_id}")
                    except Exception as e:
                        logger.error(f"Failed to cancel one-sided ask: {e}")
                state.ask_cooldown_until = time.time() + 60

    def _place_unwind_orders(self, state: MarketState, bid_price: float, ask_price: float,
                              yes_pos, no_pos):
        """Place orders only to close existing positions (unwind mode).

        In unwind mode:
        - Only SELL Yes if holding Yes
        - Only SELL No if holding No
        - Don't open new positions
        - Use slightly wider spread to ensure fills
        """
        market = state.market
        min_s = market.rewards.min_size

        logger.info(f"[UNWIND MODE] {market.question[:30]}")

        # Check if positions are now closed
        has_yes = yes_pos and yes_pos.size >= min_s
        has_no = no_pos and no_pos.size >= min_s

        if not has_yes and not has_no:
            # All positions closed - can exit
            logger.info(f"[UNWIND] All positions closed, ready to exit")
            state.unwind_only = False
            if state.marked_for_exit:
                state.marked_for_exit = False
                # Will be cleaned up on next scan
            return

        # Check position P&L - force close if loss is too large
        mid = state.orderbook.midpoint if state.orderbook else 0.5
        position_pnl = 0.0
        position_value = 0.0
        if yes_pos and yes_pos.size > 0:
            position_pnl += yes_pos.size * (mid - yes_pos.avg_cost)
            position_value += yes_pos.size * mid
        if no_pos and no_pos.size > 0:
            position_pnl += no_pos.size * ((1 - mid) - no_pos.avg_cost)
            position_value += no_pos.size * (1 - mid)

        pnl_pct = position_pnl / position_value if position_value > 0 else 0
        urgent_loss = pnl_pct < -self.cfg.unwind_urgent_loss_pct  # Default -5%

        # Check how long we've been in unwind mode
        unwind_hours = (time.time() - state.unwind_start_time) / 3600 if state.unwind_start_time else 0

        # After 4 hours, accept wider spreads; after 8 hours or urgent loss, accept any price
        if urgent_loss:
            spread_adjust = 0.03  # 3 cents wider - just get out
            logger.warning(f"[UNWIND] URGENT LOSS {pnl_pct:.1%} - accepting any price")
        elif unwind_hours > 8:
            spread_adjust = 0.02  # 2 cents wider
            logger.warning(f"[UNWIND] {unwind_hours:.1f}h elapsed - urgent mode")
        elif unwind_hours > 4:
            spread_adjust = 0.01  # 1 cent wider
        else:
            spread_adjust = 0.0

        if not self.trader:
            # Dry run
            if has_yes:
                logger.info(f"[DRY UNWIND] SELL Yes {int(yes_pos.size)}@{ask_price - spread_adjust:.3f}")
            if has_no:
                logger.info(f"[DRY UNWIND] SELL No {int(no_pos.size)}@{1 - bid_price + spread_adjust:.3f}")
            return

        # Place SELL Yes order (closes Yes position)
        if has_yes:
            sell_price = round(ask_price - spread_adjust, 3)  # Slightly lower to get filled
            sell_size = min(int(yes_pos.size), min_s)
            try:
                result = self.trader.create_limit_order(
                    yes_pos.token_id, "SELL", sell_price, sell_size,
                )
                order_id = result.get("orderID", result.get("id", "unknown"))
                self.db.record_order(
                    order_id, market.condition_id, yes_pos.token_id,
                    "SELL", sell_price, sell_size,
                )
                state.active_orders.append(ActiveOrder(
                    order_id=order_id,
                    token_id=yes_pos.token_id,
                    side=Side.SELL,
                    price=sell_price,
                    size=sell_size,
                    market_condition_id=market.condition_id,
                    is_sell_yes=True,
                ))
                logger.info(f"[UNWIND] SELL Yes {sell_size}@{sell_price}")
            except Exception as e:
                logger.error(f"[UNWIND] Failed to sell Yes: {e}")

        # Place SELL No order (closes No position)
        if has_no:
            # SELL No at (1 - bid_price) equivalent
            sell_price = round(1 - bid_price + spread_adjust, 3)
            sell_size = min(int(no_pos.size), min_s)
            try:
                result = self.trader.create_limit_order(
                    no_pos.token_id, "SELL", sell_price, sell_size,
                )
                order_id = result.get("orderID", result.get("id", "unknown"))
                self.db.record_order(
                    order_id, market.condition_id, no_pos.token_id,
                    "BUY", bid_price, sell_size,  # Equivalent bid side
                )
                state.active_orders.append(ActiveOrder(
                    order_id=order_id,
                    token_id=no_pos.token_id,
                    side=Side.BUY,
                    price=bid_price,
                    size=sell_size,
                    market_condition_id=market.condition_id,
                    is_sell_no=True,
                ))
                logger.info(f"[UNWIND] SELL No {sell_size}@{sell_price}")
            except Exception as e:
                logger.error(f"[UNWIND] Failed to sell No: {e}")

    def _check_inventory_limits(self, state: MarketState) -> tuple[bool, bool]:
        """Check if bid/ask are allowed given inventory limits.

        Returns (bid_ok, ask_ok).

        Rules:
        1. Need USDC or inventory to place orders on a side
        2. HARD CAP: stop BUYING more when inventory exceeds max_inventory_ratio
           Selling inventory is always allowed (recovers USDC)
        """
        yes_size = 0.0
        no_size = 0.0
        for pos in state.positions:
            if pos.outcome == "Yes":
                yes_size = pos.size
            elif pos.outcome == "No":
                no_size = pos.size

        bid_ok = True
        ask_ok = True
        has_capital = state.allocated_capital > 0

        # Bid: need USDC (to BUY Yes) or No tokens (to SELL No)
        if not has_capital and no_size <= 0:
            bid_ok = False

        # Ask: need USDC (to BUY No) or Yes tokens (to SELL Yes)
        if not has_capital and yes_size <= 0:
            ask_ok = False

        # HARD CAP: prevent accumulating too much on one side
        max_ratio = self.risk_cfg.max_inventory_ratio  # 0.6
        emergency = self.risk_cfg.emergency_inventory   # 0.8
        cap = max(state.allocated_capital, 50)  # fallback minimum
        min_size = state.market.rewards.min_size  # Minimum order size for rewards

        mid = state.orderbook.midpoint if state.orderbook else 0.5
        yes_value = yes_size * mid
        no_value = no_size * (1 - mid)

        # If Yes too large: block BUY Yes (bid), but allow SELL No if have enough
        # CRITICAL: Only allow bid if we have enough No to actually SELL,
        # otherwise _place_orders will fall through to BUY Yes
        if yes_value > cap * max_ratio:
            if no_size < min_size:
                bid_ok = False
                logger.warning(
                    f"Inventory cap: Yes=${yes_value:.1f} > {max_ratio:.0%} of ${cap:.0f}, "
                    f"No={no_size:.0f} < min_size={min_size} — blocking BUY Yes"
                )
            else:
                logger.info(
                    f"Yes=${yes_value:.1f} high but can SELL No={no_size:.0f}"
                )
        # If No too large: block BUY No (ask), but allow SELL Yes if have enough
        if no_value > cap * max_ratio:
            if yes_size < min_size:
                ask_ok = False
                logger.warning(
                    f"Inventory cap: No=${no_value:.1f} > {max_ratio:.0%} of ${cap:.0f}, "
                    f"Yes={yes_size:.0f} < min_size={min_size} — blocking BUY No"
                )
            else:
                logger.info(
                    f"No=${no_value:.1f} high but can SELL Yes={yes_size:.0f}"
                )

        # EMERGENCY: force sell-only
        if yes_value > cap * emergency:
            bid_ok = False
            logger.warning(f"EMERGENCY: Yes=${yes_value:.1f} — sell-only mode")
        if no_value > cap * emergency:
            ask_ok = False
            logger.warning(f"EMERGENCY: No=${no_value:.1f} — sell-only mode")

        if not bid_ok and not ask_ok:
            logger.info("Both sides blocked — no orders")

        return bid_ok, ask_ok

    def _sync_fills(self, state: MarketState):
        """Poll for new fills and update positions + allocated_capital."""
        market = state.market

        # In dry-run mode, no real fills to sync
        if not self.trader:
            return

        # Nothing to check if we have no active orders
        if not state.active_orders:
            return

        # Check which of our active orders have been filled
        try:
            open_orders = self.trader.get_open_orders()
            open_ids = {o.get("id", o.get("orderID", "")) for o in open_orders}
        except Exception as e:
            logger.debug(f"Failed to fetch open orders: {e}")
            return

        filled_orders = []
        remaining_orders = []
        for order in state.active_orders:
            if order.order_id in open_ids:
                remaining_orders.append(order)
            else:
                filled_orders.append(order)

        if not filled_orders:
            return

        state.active_orders = remaining_orders

        # Verify fills via authenticated trade history to avoid false positives
        # (orders can disappear from open_orders due to cancellation/expiry too)
        # Trade data has nested maker_orders[] with order_id field
        try:
            recent_trades = self.trader.get_trades() if self.trader else []
        except Exception:
            recent_trades = []

        # Build a map of order_id -> actual fill price from trade history
        fill_prices: dict[str, float] = {}
        filled_order_ids: set[str] = set()
        for t in (recent_trades or []):
            trade_price = float(t.get("price", 0))
            for mo in t.get("maker_orders", []):
                oid = mo.get("order_id", "")
                filled_order_ids.add(oid)
                # Use maker_order price if available, otherwise trade price
                fill_prices[oid] = float(mo.get("price", trade_price))
            # Also check top-level taker_order_id
            taker_id = t.get("taker_order_id", "")
            filled_order_ids.add(taker_id)
            fill_prices[taker_id] = trade_price

        # Calculate orderbook age for slippage tracking
        ob_age_sec = 0.0
        if state.orderbook and state.orderbook.timestamp > 0:
            ob_age_sec = time.time() - state.orderbook.timestamp

        for order in filled_orders:
            is_confirmed = order.order_id in filled_order_ids

            if not is_confirmed:
                # Order disappeared but no matching trade — likely cancelled/expired
                logger.info(
                    f"Order {order.order_id} disappeared (not in trades) — "
                    f"treating as cancelled, not filled"
                )
                self.db.mark_order_cancelled(order.order_id)
                continue

            # Get actual fill price and calculate slippage
            expected_price = order.price
            actual_price = fill_prices.get(order.order_id, order.price)
            slippage_cents = (actual_price - expected_price) * 100

            # For buys, positive slippage is bad (paid more than expected)
            # For sells, negative slippage is bad (received less than expected)
            is_adverse_slippage = (
                (order.side == Side.BUY and slippage_cents > 0) or
                (order.side == Side.SELL and slippage_cents < 0)
            )

            logger.info(
                f"Fill confirmed: {order.side.value} {order.size}@{actual_price:.4f} "
                f"(expected {expected_price:.4f}, slippage {slippage_cents:.2f}c) "
                f"in {market.question[:40]}"
            )

            # Log slippage for analysis
            slog.fill_slippage(
                market=market.question[:40],
                side=order.side.value,
                expected_price=expected_price,
                actual_price=actual_price,
                slippage_cents=slippage_cents,
                size=order.size,
                orderbook_age_sec=ob_age_sec,
            )

            # Record slippage to DB for model training
            self.db.record_slippage(
                condition_id=market.condition_id,
                order_id=order.order_id,
                side=order.side.value,
                expected_price=expected_price,
                actual_price=actual_price,
                slippage_cents=slippage_cents,
                size=order.size,
                orderbook_age_sec=ob_age_sec,
            )

            # Mark order as filled in DB
            self.db.mark_order_filled(order.order_id)

            # Alert on excessive slippage
            max_slip = getattr(self.cfg, 'max_slippage_cents', 1.0)
            if abs(slippage_cents) > max_slip:
                slog.slippage_alert(
                    market=market.question[:40],
                    side=order.side.value,
                    slippage_cents=slippage_cents,
                    max_allowed=max_slip,
                    action="ALERT" if is_adverse_slippage else "INFO",
                )
                if is_adverse_slippage:
                    logger.warning(
                        f"SLIPPAGE WARNING: {abs(slippage_cents):.2f}c > {max_slip}c threshold"
                    )

            # Track order lifecycle: fill event
            self._track_order_filled(order.order_id, actual_price)

            # Schedule price trajectory tracking for adverse selection analysis
            mid_at_fill = state.orderbook.midpoint if state.orderbook else actual_price
            self.pending_trajectory_checks.append({
                "fill_id": order.order_id,
                "fill_time": time.time(),
                "market": market.question[:40],
                "condition_id": market.condition_id,
                "fill_side": order.side.value,
                "fill_price": actual_price,
                "mid_at_fill": mid_at_fill,
                "checks_done": [],  # Will track 10s, 30s, 60s, 300s
            })

            # 更新 VPIN
            cid = market.condition_id
            if cid in self.vpin_calculators:
                mid_at_fill = state.orderbook.midpoint if state.orderbook else 0.5
                self.vpin_calculators[cid].update(order.size, order.price, mid_at_fill)

            # 记录持仓入场时间
            if order.token_id not in self.position_entry_times:
                self.position_entry_times[order.token_id] = time.time()

            # Compute spread capture: how far our fill was from mid
            mid_at_fill = state.orderbook.midpoint if state.orderbook else 0
            if order.side == Side.BUY:
                # Bought Yes below mid → capture = (mid - price) * size
                spread_capture = (mid_at_fill - order.price) * order.size
                state.pnl.n_buy_fills += 1
                state.pnl.total_buy_value += order.price * order.size
            else:
                # Bought No at (1 - ask_price), equivalent to selling Yes above mid
                # capture = (ask_price - mid) * size
                spread_capture = (order.price - mid_at_fill) * order.size
                state.pnl.n_sell_fills += 1
                no_price = round(1 - order.price, 3)
                state.pnl.total_sell_value += no_price * order.size
            state.pnl.spread_income += spread_capture
            state.pnl.n_fills += 1
            state.pnl.gas_cost += 0.005  # ~$0.005 per fill on Polygon

            # Structured log for strategy analysis
            yes_inv = sum(p.size for p in state.positions if p.outcome == "Yes")
            slog.fill(
                market=market.question[:40],
                side=order.side.value,
                price=order.price,
                size=order.size,
                mid_at_fill=mid_at_fill,
                spread_at_fill=state.orderbook.spread_cents if state.orderbook else 0,
                inventory_after=yes_inv + (order.size if order.side == Side.BUY else -order.size),
            )

            # Record fill in DB
            self.db.record_fill(
                order.order_id, market.condition_id, order.token_id,
                order.side.value, order.price, order.size,
            )

            # Update positions based on what actually happened
            is_sell_yes = getattr(order, 'is_sell_yes', False)
            is_sell_no = getattr(order, 'is_sell_no', False)

            if order.side == Side.BUY and is_sell_no:
                # Bid filled via SELL No: sold No tokens, got USDC back
                no_sell_price = round(1 - order.price, 3)
                revenue = order.size * no_sell_price
                state.allocated_capital += revenue
                self._reduce_position(state, "No", order.size)
                logger.info(f"SELL No filled: +${revenue:.2f} USDC recovered")
            elif order.side == Side.BUY:
                # Bid filled via BUY Yes: spent USDC, got Yes tokens
                cost = order.size * order.price
                state.allocated_capital -= cost
                self._update_position(state, "Yes", order.token_id, order.size, order.price)
            elif order.side == Side.SELL and is_sell_yes:
                # Ask filled via SELL Yes: sold Yes tokens, got USDC back
                revenue = order.size * order.price
                state.allocated_capital += revenue
                self._reduce_position(state, "Yes", order.size)
                logger.info(f"SELL Yes filled: +${revenue:.2f} USDC recovered")
            elif order.side == Side.SELL:
                # Ask filled via BUY No: spent USDC, got No tokens
                no_price = round(1 - order.price, 3)
                cost = order.size * no_price
                state.allocated_capital -= cost
                self._update_position(state, "No", order.token_id, order.size, no_price)

            # AUTO-REVERSE: immediately place a sell order to unwind the position
            # This prevents single-side accumulation
            self._place_reverse_order(state, order, is_sell_yes, is_sell_no)

            # Post-fill cooldown: pause the filled side
            cooldown = self.cfg.post_fill_cooldown_sec
            fill_time = time.time()
            state.last_fill_time = fill_time
            if order.side == Side.BUY:
                state.bid_cooldown_until = fill_time + cooldown
                logger.info(f"Bid cooldown set for {cooldown}s after fill")
                slog.cooldown_start(
                    market=market.question[:40], side="BUY",
                    duration_sec=cooldown,
                )
            elif order.side == Side.SELL:
                state.ask_cooldown_until = fill_time + cooldown
                logger.info(f"Ask cooldown set for {cooldown}s after fill")
                slog.cooldown_start(
                    market=market.question[:40], side="SELL",
                    duration_sec=cooldown,
                )

            # Track for recent_fills (used by GM spread)
            state.recent_fills.append({
                "side": order.side.value,
                "size": order.size,
                "price": order.price,
            })
            if len(state.recent_fills) > self.cfg.gm_lookback_fills:
                state.recent_fills = state.recent_fills[-self.cfg.gm_lookback_fills:]

        # 有成交确认后，强制从 API 同步真实持仓（覆盖本地计算）
        if filled_orders:
            self.force_position_sync(state)

    def _update_position(
        self, state: MarketState, outcome: str, token_id: str,
        qty: float, price: float,
    ):
        """Add to a position (from a buy fill)."""
        for pos in state.positions:
            if pos.outcome == outcome:
                total_cost = pos.avg_cost * pos.size + price * qty
                pos.size += qty
                pos.avg_cost = total_cost / pos.size if pos.size > 0 else 0
                self.db.update_position(token_id, state.market.condition_id, outcome, pos.size, pos.avg_cost)
                return
        # New position
        pos = Position(token_id=token_id, outcome=outcome, size=qty, avg_cost=price)
        state.positions.append(pos)
        self.db.update_position(token_id, state.market.condition_id, outcome, qty, price)

    def _reduce_position(self, state: MarketState, outcome: str, qty: float):
        """Reduce a position (from a sell fill)."""
        for pos in state.positions:
            if pos.outcome == outcome:
                pos.size = max(0, pos.size - qty)
                self.db.update_position(pos.token_id, state.market.condition_id, outcome, pos.size, pos.avg_cost)
                return
        logger.warning(f"Sell fill for {outcome} but no tracked position — position tracking may be out of sync")

    def _place_reverse_order(
        self, state: MarketState, filled_order: ActiveOrder,
        is_sell_yes: bool, is_sell_no: bool,
    ):
        """After a BUY fill, immediately place a SELL order to unwind.

        改进：使用 UnwindOptimizer 计算最优平仓价格，而不是固定 +2%
        """
        if not self.trader:
            return

        # Only reverse BUY fills (not SELL fills which are already unwinding)
        if is_sell_yes or is_sell_no:
            return  # Already a sell — no reverse needed

        market = state.market
        size = filled_order.size
        price = filled_order.price
        ob = state.orderbook

        if not ob:
            # 没有订单簿，用默认 +2%
            sell_price = round(price * 1.02, 3)
        else:
            # 使用 UnwindOptimizer 计算最优价格
            if filled_order.side == Side.BUY:
                # 买入 YES
                unwind_price, strategy, urgency = self.unwind_optimizer.compute_unwind_price(
                    outcome="Yes",
                    avg_cost=price,
                    current_mid=ob.midpoint,
                    hours_held=0,  # 刚买入
                    orderbook=ob,
                )
                sell_price = unwind_price
            else:
                # 买入 NO (通过 ask 侧)
                no_buy_price = round(1 - price, 3)
                unwind_price, strategy, urgency = self.unwind_optimizer.compute_unwind_price(
                    outcome="No",
                    avg_cost=no_buy_price,
                    current_mid=ob.midpoint,
                    hours_held=0,
                    orderbook=ob,
                )
                sell_price = unwind_price

        sell_price = min(sell_price, 0.999)
        sell_price = max(sell_price, 0.001)

        if filled_order.side == Side.BUY:
            # Bought Yes tokens → SELL Yes at computed price
            token = market.yes_token
            if not token or size < 1:
                return
            try:
                result = self.trader.create_limit_order(
                    token.token_id, "SELL", sell_price, int(size),
                )
                order_id = result.get("orderID", result.get("id", "unknown"))
                profit_pct = (sell_price - price) / price * 100 if price > 0 else 0
                logger.info(
                    f"AUTO-REVERSE: SELL Yes {int(size)}@{sell_price} "
                    f"(bought @{price}, target +{profit_pct:.1f}%)"
                )
            except Exception as e:
                logger.error(f"Auto-reverse SELL Yes failed: {e}")

        elif filled_order.side == Side.SELL:
            # Bought No tokens → SELL No at computed price
            no_buy_price = round(1 - price, 3)

            # Find the No token
            for pos in state.positions:
                if pos.outcome == "No" and pos.size > 0:
                    try:
                        result = self.trader.create_limit_order(
                            pos.token_id, "SELL", sell_price, int(size),
                        )
                        order_id = result.get("orderID", result.get("id", "unknown"))
                        profit_pct = (sell_price - no_buy_price) / no_buy_price * 100 if no_buy_price > 0 else 0
                        logger.info(
                            f"AUTO-REVERSE: SELL No {int(size)}@{sell_price} "
                            f"(bought @{no_buy_price}, target +{profit_pct:.1f}%)"
                        )
                    except Exception as e:
                        logger.error(f"Auto-reverse SELL No failed: {e}")
                    break

    def _maybe_take_profit(self, state: MarketState):
        """检查持仓，执行止盈或止损。

        改进：
        1. 使用 UnwindOptimizer 计算最优平仓价格
        2. 添加止损逻辑
        3. 考虑持仓时间
        """
        if not state.positions or not state.orderbook or not self.trader:
            return

        ob = state.orderbook
        mid = ob.midpoint
        cid = state.market.condition_id

        # 获取动量信息
        detector = self.eat_detectors.get(cid)
        if detector:
            momentum, momentum_dir = detector.detect_momentum()
        else:
            momentum, momentum_dir = 0, "neutral"

        for pos in state.positions:
            if pos.size <= 0 or pos.avg_cost <= 0:
                continue

            # 计算持仓时间
            entry_time = self.position_entry_times.get(pos.token_id, time.time())
            hours_held = (time.time() - entry_time) / 3600

            if pos.outcome == "Yes":
                current_price = mid
                pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost
            else:
                current_price = 1 - mid
                pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost

            # 使用 UnwindOptimizer 计算最优平仓价格
            unwind_price, strategy, urgency = self.unwind_optimizer.compute_unwind_price(
                outcome=pos.outcome,
                avg_cost=pos.avg_cost,
                current_mid=mid,
                hours_held=hours_held,
                orderbook=ob,
            )

            # 检查是否应该市价卖出
            should_market = self.unwind_optimizer.should_market_sell(
                pnl_pct=pnl_pct,
                hours_held=hours_held,
                momentum=momentum,
                momentum_direction=momentum_dir,
                my_side=pos.outcome,
            )

            if should_market:
                # 市价卖出 - 使用 best_bid
                if pos.outcome == "Yes":
                    market_price = ob.bids[0].price if ob.bids else mid * 0.98
                else:
                    market_price = 1 - ob.asks[0].price if ob.asks else current_price * 0.98

                logger.warning(
                    f"MARKET SELL {pos.outcome}: size={pos.size:.0f} cost={pos.avg_cost:.3f} "
                    f"pnl={pnl_pct:.1%} hours={hours_held:.1f} momentum={momentum_dir}"
                )
                try:
                    self.trader.create_limit_order(
                        pos.token_id, "SELL", round(market_price, 3), int(pos.size),
                    )
                    slog.stop_loss(
                        market=state.market.question[:40],
                        outcome=pos.outcome,
                        size=pos.size,
                        avg_cost=pos.avg_cost,
                        sell_price=market_price,
                        pnl_pct=pnl_pct,
                        reason="market_sell",
                    )
                except Exception as e:
                    logger.error(f"Market sell failed: {e}")
                continue

            # 根据策略和紧急程度决定是否下单
            if urgency >= 2 or strategy == "urgent_stop":
                # 紧急止损
                logger.warning(
                    f"URGENT STOP {pos.outcome}: size={pos.size:.0f} cost={pos.avg_cost:.3f} "
                    f"price={unwind_price:.3f} pnl={pnl_pct:.1%}"
                )
                try:
                    self.trader.create_limit_order(
                        pos.token_id, "SELL", unwind_price, int(pos.size),
                    )
                    slog.stop_loss(
                        market=state.market.question[:40],
                        outcome=pos.outcome,
                        size=pos.size,
                        avg_cost=pos.avg_cost,
                        sell_price=unwind_price,
                        pnl_pct=pnl_pct,
                        reason=strategy,
                    )
                except Exception as e:
                    logger.error(f"Urgent stop failed: {e}")

            elif urgency >= 1 or strategy in ("cut_loss", "breakeven"):
                # 止损或保本
                logger.info(
                    f"{strategy.upper()} {pos.outcome}: size={pos.size:.0f} cost={pos.avg_cost:.3f} "
                    f"target={unwind_price:.3f} pnl={pnl_pct:.1%} hours={hours_held:.1f}"
                )
                try:
                    self.trader.create_limit_order(
                        pos.token_id, "SELL", unwind_price, int(pos.size),
                    )
                except Exception as e:
                    logger.error(f"{strategy} order failed: {e}")

            elif pnl_pct >= 0.015:
                # 有利润，挂止盈单
                logger.info(
                    f"Take-profit {pos.outcome}: size={pos.size:.0f} cost={pos.avg_cost:.3f} "
                    f"target={unwind_price:.3f} pnl={pnl_pct:.1%}"
                )
                try:
                    self.trader.create_limit_order(
                        pos.token_id, "SELL", unwind_price, int(pos.size),
                    )
                except Exception as e:
                    logger.error(f"Take-profit failed: {e}")

    def _accrue_reward(self, state: MarketState):
        """Estimate reward accrual for this tick based on Q share.

        daily_rate * (your_q / (total_q + your_q)) / ticks_per_day
        """
        if not state.active_orders or not state.orderbook:
            return
        if state.total_q_min <= 0 or state.market.rewards.daily_rate <= 0:
            return

        # Estimate your Q from active orders
        from .scanner import compute_q_score
        max_spread = state.market.rewards.max_spread
        mid = state.orderbook.midpoint
        your_q = 0.0
        for order in state.active_orders:
            dist_c = abs(order.price - mid) * 100
            your_q += compute_q_score(order.size, dist_c, max_spread)

        if your_q <= 0:
            return

        your_share = your_q / (state.total_q_min + your_q)
        daily_rate = state.market.rewards.daily_rate

        # Convert to per-tick: refresh_sec / 86400
        tick_sec = self.cfg.order_refresh_sec
        per_tick = daily_rate * your_share * (tick_sec / 86400)
        state.pnl.reward_income += per_tick

        # Log Q score update for analysis (sample every 5 minutes to reduce volume)
        if int(time.time()) % 300 < tick_sec:
            est_hourly = daily_rate * your_share / 24
            slog.q_score_update(
                market=state.market.question[:40],
                your_q=your_q,
                total_q=state.total_q_min,
                q_share_pct=your_share * 100,
                reward_rate=daily_rate,
                est_hourly_reward=est_hourly,
            )

    def emergency_cancel_all(self, state: MarketState):
        """Cancel all orders immediately (risk event)."""
        logger.warning(f"EMERGENCY: Cancelling all orders for {state.market.question[:40]}")
        self._cancel_existing_orders(state)

    def save_cfr_state(self):
        """Persist CFR state to database."""
        for key, regret, strat, rounds in self.cfr.save_state():
            self.db.update_cfr_state(key, regret, strat, rounds)
