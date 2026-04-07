"""Risk manager - drawdown control, daily limits, settlement protection."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .config import RiskConfig
from .db import Database
from .strategy_log import slog
from .types import MarketState

logger = logging.getLogger("mantis.risk")


class RiskAction:
    """Risk decision returned by check()."""
    CONTINUE = "continue"
    WIDEN_SPREAD = "widen_spread"
    PAUSE_SIDE = "pause_side"
    EMERGENCY_WITHDRAW = "emergency_withdraw"
    MIGRATE = "migrate"


class RiskManager:
    def __init__(self, config: RiskConfig, db: Database, total_capital: float):
        self.cfg = config
        self.db = db
        self.total_capital = total_capital
        self.peak_value = total_capital
        self.daily_start_value = total_capital
        self.daily_pnl = 0.0
        self._paused_until: float = 0.0

    def check(self, state: MarketState) -> str:
        """Run all risk checks. Returns the most severe action needed."""
        if time.time() < self._paused_until:
            return RiskAction.EMERGENCY_WITHDRAW

        actions = [
            self._check_drawdown(state),
            self._check_daily_loss(state),
            self._check_settlement(state),
            self._check_ecosystem(state),
            self._check_volatility(state),
        ]

        # Return most severe
        priority = {
            RiskAction.EMERGENCY_WITHDRAW: 4,
            RiskAction.MIGRATE: 3,
            RiskAction.PAUSE_SIDE: 2,
            RiskAction.WIDEN_SPREAD: 1,
            RiskAction.CONTINUE: 0,
        }

        return max(actions, key=lambda a: priority.get(a, 0))

    def _current_value(self, state: MarketState) -> float:
        """Calculate current portfolio value for this market."""
        value = state.allocated_capital  # cash portion

        if not state.orderbook:
            # No orderbook yet — use cost basis to avoid fake mid=0.5 valuations
            for pos in state.positions:
                value += pos.size * pos.avg_cost
            return value

        mid = state.orderbook.midpoint
        for pos in state.positions:
            if pos.outcome == "Yes":
                value += pos.size * mid
            elif pos.outcome == "No":
                value += pos.size * (1 - mid)

        return value

    def _check_drawdown(self, state: MarketState) -> str:
        """Check max drawdown from per-market peak."""
        if not state.orderbook:
            return RiskAction.CONTINUE  # skip until we have real prices

        current = self._current_value(state)

        # Track peak per market (not global peak which mixes total capital
        # with single-market values)
        if current > state.peak_value:
            state.peak_value = current

        if state.peak_value <= 0:
            return RiskAction.CONTINUE

        drawdown = (state.peak_value - current) / state.peak_value

        if drawdown >= self.cfg.max_drawdown:
            self.db.record_risk_event(
                "max_drawdown",
                f"Drawdown {drawdown:.1%} from peak ${state.peak_value:.2f}",
            )
            logger.warning(
                f"MAX DRAWDOWN {drawdown:.1%} hit! "
                f"Peak=${state.peak_value:.2f} Current=${current:.2f}"
            )
            slog.risk_trigger(
                market=state.market.question[:40], rule="max_drawdown",
                action="emergency_withdraw",
                drawdown_pct=drawdown, peak=state.peak_value, current=current,
            )
            # No global pause — drawdown is per-market and persistent
            # (peak stays high, so this market keeps getting blocked)
            return RiskAction.EMERGENCY_WITHDRAW

        if drawdown >= self.cfg.max_drawdown * 0.7:
            logger.info(f"Drawdown warning: {drawdown:.1%}")
            slog.risk_trigger(
                market=state.market.question[:40], rule="drawdown_warning",
                action="widen_spread", drawdown_pct=drawdown,
            )
            return RiskAction.WIDEN_SPREAD

        return RiskAction.CONTINUE

    def _check_daily_loss(self, state: MarketState) -> str:
        """Check daily loss limit (uses aggregate daily_pnl from update_daily_pnl)."""
        if self.daily_pnl <= -self.cfg.daily_loss_limit:
            self.db.record_risk_event(
                "daily_loss_limit",
                f"Daily loss ${abs(self.daily_pnl):.2f} exceeds limit ${self.cfg.daily_loss_limit}",
            )
            logger.warning(f"DAILY LOSS LIMIT hit: ${self.daily_pnl:.2f}")
            return RiskAction.EMERGENCY_WITHDRAW

        return RiskAction.CONTINUE

    def _check_settlement(self, state: MarketState) -> str:
        """Check if market is approaching settlement."""
        end_date = state.market.end_date
        if not end_date:
            return RiskAction.CONTINUE

        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_left = (end_dt - now).total_seconds() / 3600

            if hours_left <= self.cfg.settlement_buffer_hours:
                logger.warning(
                    f"Market settles in {hours_left:.0f}h - withdrawing"
                )
                self.db.record_risk_event(
                    "settlement_approaching",
                    f"{hours_left:.0f} hours to settlement",
                )
                return RiskAction.EMERGENCY_WITHDRAW

            if hours_left <= self.cfg.settlement_buffer_hours * 1.5:
                return RiskAction.WIDEN_SPREAD

        except (ValueError, TypeError):
            pass

        return RiskAction.CONTINUE

    def _check_volatility(self, state: MarketState) -> str:
        """Pause quoting when 3h volatility exceeds threshold."""
        if state.volatility_3h > self.cfg.vol_pause_threshold:
            logger.warning(
                f"High volatility {state.volatility_3h:.1%} > "
                f"{self.cfg.vol_pause_threshold:.1%} — widening spread"
            )
            slog.risk_trigger(
                market=state.market.question[:40], rule="high_volatility",
                action="widen_spread",
                vol_3h=state.volatility_3h,
                threshold=self.cfg.vol_pause_threshold,
            )
            state.spread_multiplier = max(state.spread_multiplier, 2.0)
            return RiskAction.WIDEN_SPREAD
        return RiskAction.CONTINUE

    def _check_ecosystem(self, state: MarketState) -> str:
        """Check if market ecosystem has deteriorated."""
        # This relies on the scanner's ecosystem detection
        # If reward_per_q has dropped significantly, suggest migration
        if state.reward_per_q > 0 and state.reward_per_q < 0.01:
            return RiskAction.MIGRATE

        return RiskAction.CONTINUE

    def update_daily_pnl(self, states: list) -> float:
        """Compute aggregate daily P&L across all markets."""
        # Only use states with real orderbook data to avoid fake valuations
        priced_states = [s for s in states if s.orderbook]
        if not priced_states:
            return self.daily_pnl  # no real prices yet, skip

        total_value = sum(self._current_value(s) for s in priced_states)
        # On first call after restart, calibrate start value to actual portfolio
        if not hasattr(self, '_calibrated') or not self._calibrated:
            if total_value > 0:
                self.daily_start_value = total_value
                self._calibrated = True
                logger.info(f"Calibrated daily start value to ${total_value:.2f}")
        self.daily_pnl = total_value - self.daily_start_value
        return self.daily_pnl

    def reset_daily(self, current_value: float):
        """Call at start of each day to reset daily counters."""
        self.daily_start_value = current_value
        self.daily_pnl = 0.0
        logger.info(f"Daily reset. Starting value: ${current_value:.2f}")

    @property
    def is_paused(self) -> bool:
        return time.time() < self._paused_until
