"""Order engine - core market making logic with game theory pricing."""
from __future__ import annotations

import logging
import time

from .config import MantisConfig
from .db import Database
from .game_theory import (
    CFREngine,
    detect_ecosystem_state,
    glosten_milgrom_min_spread,
    markov_price_drift,
    nash_optimal_spread,
)
from .polymarket_client import PolymarketClient, PolymarketTrader
from .strategy_log import slog
from .types import ActiveOrder, MarketState, Orderbook, Position, Side

logger = logging.getLogger("mantis.engine")


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

    def tick(self, state: MarketState) -> MarketState:
        """Main loop tick for a single market. Called every order_refresh_sec.

        1. Fetch latest orderbook
        2. Record price history
        3. Compute quotes using game theory stack
        4. Compare with existing orders
        5. Reprice if needed
        """
        market = state.market
        token = market.yes_token
        if not token:
            return state

        # 1. Fetch orderbook
        try:
            ob = self.client.fetch_orderbook(token.token_id)
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return state
        state.orderbook = ob

        if not ob.bids or not ob.asks:
            logger.warning(f"Empty orderbook for {market.question[:40]}")
            return state

        # 1a. On first tick, cancel stale exchange orders from previous runs
        if not state.orders_cleaned_up:
            self._cleanup_stale_orders(state)
            state.orders_cleaned_up = True

        # 1b. Sync fills and update positions
        self._sync_fills(state)

        # 1c. Check take-profit on existing inventory
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
        """Compute bid/ask prices using the game theory stack."""
        ob = state.orderbook
        assert ob is not None
        mid = ob.midpoint
        market = state.market

        # Reward farming mode: skip game theory, place tight
        if self.cfg.reward_mode:
            return self._compute_reward_quotes(state)

        # Layer 1: Markov price drift
        drift = markov_price_drift(state.recent_prices)
        drift = max(-0.01, min(0.01, drift))  # cap at 1 cent

        # Layer 2: Glosten-Milgrom minimum spread
        gm_min = glosten_milgrom_min_spread(
            state.recent_fills,
            informed_ratio=0.15,
            expected_jump=0.05,
        )

        # Layer 3: Nash optimal spread
        nash_spread = nash_optimal_spread(ob, gm_min, aggression=0.9)

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

        # Apply risk-driven spread widening
        half *= state.spread_multiplier

        # Layer 4: Inventory skew adjustment
        bid_adj, ask_adj = self._inventory_adjustment(state)

        # Compute final prices
        bid_price = round(mid + drift - half + bid_adj, 3)
        ask_price = round(mid + drift + half + ask_adj, 3)

        # Sanity: bid must be positive, ask must be < 1
        bid_price = max(0.001, bid_price)
        ask_price = min(0.999, ask_price)

        # Bid must be < ask
        if bid_price >= ask_price:
            bid_price = round(mid - self.cfg.min_half_spread, 3)
            ask_price = round(mid + self.cfg.min_half_spread, 3)

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

    def _place_orders(self, state: MarketState, bid_price: float, ask_price: float):
        """Place new bid and ask orders."""
        market = state.market
        token = market.yes_token
        if not token:
            return

        # Check inventory — sides with inventory don't need USDC
        yes_pos = next((p for p in state.positions if p.outcome == "Yes" and p.size > 0), None)
        no_pos = next((p for p in state.positions if p.outcome == "No" and p.size > 0), None)

        # Smart capital split: if one side has inventory (will SELL),
        # that side needs 0 USDC, so give all capital to the other side
        total_capital = max(0, state.allocated_capital)
        bid_needs_capital = not (no_pos and no_pos.size > 0)   # SELL No doesn't need USDC
        ask_needs_capital = not (yes_pos and yes_pos.size > 0)  # SELL Yes doesn't need USDC

        if bid_needs_capital and ask_needs_capital:
            capital_per_side = total_capital / 2
        elif bid_needs_capital:
            capital_per_side = total_capital  # all capital to bid, ask uses inventory
        elif ask_needs_capital:
            capital_per_side = total_capital  # all capital to ask, bid uses inventory
        else:
            capital_per_side = 0  # both sides use inventory

        has_inventory = yes_pos is not None or no_pos is not None

        if total_capital <= 0 and not has_inventory:
            logger.info("No capital and no inventory — skipping order placement")
            return

        min_s = market.rewards.min_size
        if self.cfg.reward_mode:
            bid_cost = min_s * bid_price
            ask_cost = min_s * (1 - ask_price)
            bid_size = min_s if (not bid_needs_capital) or bid_cost <= capital_per_side else 0
            ask_size = min_s if (not ask_needs_capital) or ask_cost <= capital_per_side else 0
        else:
            if capital_per_side > 0:
                affordable_bid = int(capital_per_side / bid_price)
                affordable_ask = int(capital_per_side / (1 - ask_price))
                bid_size = max(affordable_bid, min_s) if affordable_bid >= min_s else 0
                ask_size = max(affordable_ask, min_s) if affordable_ask >= min_s else 0
            else:
                bid_size = 0
                ask_size = 0

        # When selling inventory, use position size (don't need USDC capital)
        if bid_size == 0 and no_pos and no_pos.size > 0:
            bid_size = min(int(no_pos.size), min_s)
        if ask_size == 0 and yes_pos and yes_pos.size > 0:
            ask_size = min(int(yes_pos.size), min_s)

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
                except Exception as e:
                    logger.error(f"Failed to place bid (SELL No): {e}")
            else:
                # BUY Yes with USDC capital
                if capital_per_side > 0 and bid_size > 0:
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
                    except Exception as e:
                        logger.error(f"Failed to place bid (BUY Yes): {e}")

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
                except Exception as e:
                    logger.error(f"Failed to place ask (SELL Yes): {e}")
            else:
                # BUY No (equivalent to SELL Yes side) with USDC capital
                if capital_per_side > 0 and ask_size > 0 and no_token:
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
                    except Exception as e:
                        logger.error(f"Failed to place ask (BUY No): {e}")

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

        mid = state.orderbook.midpoint if state.orderbook else 0.5
        yes_value = yes_size * mid
        no_value = no_size * (1 - mid)

        # If Yes too large: block BUY Yes (bid), but allow SELL No
        if yes_value > cap * max_ratio and no_size <= 0:
            bid_ok = False
            logger.warning(
                f"Inventory cap: Yes=${yes_value:.1f} > {max_ratio:.0%} of ${cap:.0f}"
            )
        # If No too large: block BUY No (ask), but allow SELL Yes
        if no_value > cap * max_ratio and yes_size <= 0:
            ask_ok = False
            logger.warning(
                f"Inventory cap: No=${no_value:.1f} > {max_ratio:.0%} of ${cap:.0f}"
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
        filled_order_ids: set[str] = set()
        for t in (recent_trades or []):
            for mo in t.get("maker_orders", []):
                filled_order_ids.add(mo.get("order_id", ""))
            # Also check top-level taker_order_id
            filled_order_ids.add(t.get("taker_order_id", ""))

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

            logger.info(
                f"Fill confirmed: {order.side.value} {order.size}@{order.price} "
                f"in {market.question[:40]}"
            )

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
        """After a BUY fill, immediately place a SELL order at cost+2% to unwind.

        This is the core fix for single-side accumulation: every buy is paired
        with a sell. If the sell gets filled, we net out with profit.
        If it doesn't fill, take_profit will catch it later.
        """
        if not self.trader:
            return

        # Only reverse BUY fills (not SELL fills which are already unwinding)
        if is_sell_yes or is_sell_no:
            return  # Already a sell — no reverse needed

        market = state.market
        size = filled_order.size
        price = filled_order.price

        # Sell at cost + 2% profit
        sell_price = round(price * 1.02, 3)
        sell_price = min(sell_price, 0.999)

        if filled_order.side == Side.BUY:
            # Bought Yes tokens → SELL Yes at higher price
            token = market.yes_token
            if not token or size < 1:
                return
            try:
                result = self.trader.create_limit_order(
                    token.token_id, "SELL", sell_price, int(size),
                )
                order_id = result.get("orderID", result.get("id", "unknown"))
                logger.info(
                    f"AUTO-REVERSE: SELL Yes {int(size)}@{sell_price} "
                    f"(bought @{price}, +2% profit target)"
                )
                # Don't track in active_orders (it's a background order)
                # It will be cleaned up by _cleanup_stale_orders on restart
            except Exception as e:
                logger.error(f"Auto-reverse SELL Yes failed: {e}")

        elif filled_order.side == Side.SELL:
            # Bought No tokens → SELL No at higher price
            # filled_order.price is the ask_price (Yes side), so No was bought at 1-ask
            no_buy_price = round(1 - price, 3)
            no_sell_price = round(no_buy_price * 1.02, 3)
            no_sell_price = min(no_sell_price, 0.999)

            # Find the No token
            for pos in state.positions:
                if pos.outcome == "No" and pos.size > 0:
                    try:
                        result = self.trader.create_limit_order(
                            pos.token_id, "SELL", no_sell_price, int(size),
                        )
                        order_id = result.get("orderID", result.get("id", "unknown"))
                        logger.info(
                            f"AUTO-REVERSE: SELL No {int(size)}@{no_sell_price} "
                            f"(bought @{no_buy_price}, +2% profit target)"
                        )
                    except Exception as e:
                        logger.error(f"Auto-reverse SELL No failed: {e}")
                    break

    def _maybe_take_profit(self, state: MarketState):
        """If holding inventory with 2%+ unrealized gain, place a take-profit SELL order.

        Directly SELL the tokens we hold to recover USDC.
        """
        if not state.positions or not state.orderbook or not self.trader:
            return
        mid = state.orderbook.midpoint

        for pos in state.positions:
            if pos.size <= 0 or pos.avg_cost <= 0:
                continue

            if pos.outcome == "Yes":
                gain_pct = (mid - pos.avg_cost) / pos.avg_cost
                if gain_pct >= 0.02:
                    tp_price = round(pos.avg_cost * 1.02, 3)
                    tp_price = min(tp_price, 0.999)
                    logger.info(
                        f"Take-profit: SELL Yes avg={pos.avg_cost:.3f} mid={mid:.3f} "
                        f"gain={gain_pct:.1%} → SELL @{tp_price:.3f}"
                    )
                    try:
                        self.trader.create_limit_order(
                            pos.token_id, "SELL", tp_price, int(pos.size),
                        )
                    except Exception as e:
                        logger.error(f"Take-profit SELL Yes failed: {e}")

            elif pos.outcome == "No":
                no_mid = 1 - mid
                gain_pct = (no_mid - pos.avg_cost) / pos.avg_cost
                if gain_pct >= 0.02:
                    tp_price = round(pos.avg_cost * 1.02, 3)
                    tp_price = min(tp_price, 0.999)
                    logger.info(
                        f"Take-profit: SELL No avg={pos.avg_cost:.3f} no_mid={no_mid:.3f} "
                        f"gain={gain_pct:.1%} → SELL @{tp_price:.3f}"
                    )
                    try:
                        self.trader.create_limit_order(
                            pos.token_id, "SELL", tp_price, int(pos.size),
                        )
                    except Exception as e:
                        logger.error(f"Take-profit SELL No failed: {e}")

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

    def emergency_cancel_all(self, state: MarketState):
        """Cancel all orders immediately (risk event)."""
        logger.warning(f"EMERGENCY: Cancelling all orders for {state.market.question[:40]}")
        self._cancel_existing_orders(state)

    def save_cfr_state(self):
        """Persist CFR state to database."""
        for key, regret, strat, rounds in self.cfr.save_state():
            self.db.update_cfr_state(key, regret, strat, rounds)
