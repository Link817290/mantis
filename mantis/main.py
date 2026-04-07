"""Mantis - main entry point and scheduler."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import MantisConfig, load_config
from .db import Database
from .engine import OrderEngine
from .game_theory import detect_ecosystem_state
from .polymarket_client import PolymarketClient, PolymarketTrader
from .risk import RiskAction, RiskManager
from .scanner import MarketScanner, compute_total_q_min, estimate_your_q
from .strategy_log import slog
from .types import MarketState, Position, Side

logger = logging.getLogger("mantis")


class Mantis:
    """Main scheduler that coordinates all modules."""

    def __init__(self, config: MantisConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self._running = False

        # Initialize components
        self.client = PolymarketClient()
        self.db = Database()

        if dry_run or not config.wallet.private_key:
            self.trader = None
            if not dry_run:
                logger.warning("No private key configured - running in DRY RUN mode")
            self.dry_run = True
        else:
            self.trader = PolymarketTrader(
                config.wallet.private_key,
                funder=config.wallet.browser_address,
            )

        self.scanner = MarketScanner(self.client, config)
        self.engine = OrderEngine(self.client, self.trader, self.db, config)
        self.risk = RiskManager(config.risk, self.db, config.capital)

        # Active market states
        self.states: list[MarketState] = []

        # Proxy address for position sync
        self._proxy_address = config.wallet.browser_address or ""

        # Timing
        self._last_scan = 0.0
        self._last_quick_scan = 0.0
        self._last_daily_reset = ""
        self._last_heartbeat = 0.0
        self._heartbeat_interval = 30.0  # 每30秒发送一次心跳

        # Sync positions from chain on startup
        self._sync_positions_from_chain()

    def _sync_positions_from_chain(self):
        """Sync DB positions with real on-chain positions via data-api."""
        if not self._proxy_address:
            # Try to get address from trader
            if self.trader:
                try:
                    self.trader._ensure_client()
                    self._proxy_address = self.trader._client.get_address()
                except Exception:
                    pass
        if not self._proxy_address:
            logger.warning("No proxy address — skipping position sync")
            return

        try:
            real_positions = self.client.fetch_positions(self._proxy_address)
            logger.info(f"Chain position sync: {len(real_positions)} positions found")

            # Clear old DB positions and write fresh ones
            self.db.clear_all_positions()

            for p in real_positions:
                self.db.update_position(
                    token_id=p["asset"],
                    condition_id=p["conditionId"],
                    outcome=p.get("outcome", "Unknown"),
                    size=p["size"],
                    avg_cost=p.get("avgPrice", 0),
                )
                logger.info(
                    f"  Synced: {p.get('outcome')} x{p['size']} "
                    f"@ ${p.get('avgPrice', 0):.4f} — {p.get('title', '?')[:40]}"
                )
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    def start(self):
        """Start the main loop."""
        self._running = True
        logger.info("=" * 60)
        logger.info(f"  Mantis Market Maker v0.1")
        logger.info(f"  Capital: ${self.config.capital}")
        logger.info(f"  Max markets: {self.config.markets.max_active}")
        logger.info(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)

        slog.bot_start(
            capital=self.config.capital,
            mode="dry_run" if self.dry_run else "live",
            max_markets=self.config.markets.max_active,
        )

        # Initial scan
        self._scan_and_select()

        # Main loop
        while self._running:
            try:
                self._tick()
                time.sleep(self.config.engine.order_refresh_sec)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                slog.error(context="main_loop", message=str(e))
                time.sleep(10)

    def stop(self):
        self._running = False

    def _tick(self):
        """One iteration of the main loop."""
        now = time.time()

        # 心跳：防止订单被自动取消（每30秒）
        if self.trader and now - self._last_heartbeat >= self._heartbeat_interval:
            if self.trader.send_heartbeat():
                logger.debug("Heartbeat sent")
            else:
                logger.warning("Heartbeat failed - orders may be cancelled!")
            self._last_heartbeat = now

        # Daily reset check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_daily_reset:
            self._daily_reset(today)

        # Periodic market scan (full scan every 10 min, quick scan every 5 min)
        scan_interval = self.config.markets.scan_interval_min * 60
        quick_scan_interval = self.config.markets.quick_scan_interval_min * 60

        if now - self._last_scan >= scan_interval:
            self._scan_and_select()
        elif now - self._last_quick_scan >= quick_scan_interval:
            self._quick_opportunity_check()

        # Update aggregate daily P&L before risk checks
        if self.states:
            self.risk.update_daily_pnl(self.states)

        # Process each active market
        needs_rescan = False
        for state in self.states:
            # Risk check first
            risk_action = self.risk.check(state)

            if risk_action == RiskAction.EMERGENCY_WITHDRAW:
                self.engine.emergency_cancel_all(state)
                continue

            if risk_action == RiskAction.MIGRATE:
                logger.info(f"Risk suggests migration from {state.market.question[:40]}")
                needs_rescan = True
                continue

            if risk_action == RiskAction.WIDEN_SPREAD:
                if state.spread_multiplier != 1.5:
                    logger.info(f"Widening spread 1.5x for {state.market.question[:40]}")
                state.spread_multiplier = 1.5
            else:
                state.spread_multiplier = 1.0

            # Detect ecosystem changes
            prev_ob = state.orderbook
            state = self.engine.tick(state)

            if prev_ob and state.orderbook:
                eco = detect_ecosystem_state(state.orderbook, prev_ob)
                if eco in ("crowded", "whale_entered"):
                    logger.info(f"Ecosystem: {eco} in {state.market.question[:40]}")
                    # Update reward_per_q for risk tracking
                    if state.orderbook:
                        total_q = compute_total_q_min(
                            state.orderbook, state.market.rewards.max_spread,
                        )
                        state.total_q_min = total_q
                        old_rpq = state.reward_per_q
                        state.reward_per_q = (
                            state.market.rewards.daily_rate / total_q
                            if total_q > 0 else 0
                        )
                        # If reward dropped significantly, trigger migration check
                        if old_rpq > 0 and state.reward_per_q < old_rpq * 0.5:
                            logger.warning(
                                f"Reward dropped {old_rpq:.4f} -> {state.reward_per_q:.4f}, "
                                f"triggering migration check"
                            )
                            needs_rescan = True

            # Check if quality monitor flagged for exit
            if state.marked_for_exit and not state.unwind_only:
                logger.warning(
                    f"Quality monitor flagged {state.market.question[:30]} for migration "
                    f"(quality={state.quality_score:.2f})"
                )
                needs_rescan = True

        # Deferred rescan (outside iteration loop)
        if needs_rescan:
            self._scan_and_select()

        # Process pending price trajectory checks for adverse selection analysis
        states_by_cid = {s.market.condition_id: s for s in self.states}
        self.engine._process_trajectory_checks(states_by_cid)

        # Status log
        self._log_status()

    def _compute_available_capital(self) -> float:
        """Compute how much capital is still available (not tied up in positions).

        Checks ALL positions in DB (not just active markets) to prevent
        over-allocation after restarts.
        """
        total_invested = 0.0
        for p in self.db.get_all_positions():
            total_invested += p["size"] * p["avg_cost"]
        available = max(0, self.config.capital - total_invested)
        logger.info(
            f"Capital: ${self.config.capital:.0f} total, "
            f"${total_invested:.2f} in positions, "
            f"${available:.2f} available"
        )
        return available

    def _quick_opportunity_check(self):
        """Quick check for better market opportunities between full scans."""
        self._last_quick_scan = time.time()

        current_ids = {s.market.condition_id for s in self.states}
        if not current_ids:
            return  # No active markets, wait for full scan

        # Get current best RPC for comparison
        current_best_rpc = 0.0
        if self.states:
            for s in self.states:
                if s.reward_per_q > 0 and s.total_q_min > 0:
                    est_rpc = (s.market.rewards.daily_rate * s.reward_per_q) / (
                        s.market.rewards.min_size * 0.5 * 2
                    )
                    current_best_rpc = max(current_best_rpc, est_rpc)

        # Quick scan for opportunities
        opportunities = self.scanner.quick_scan(current_ids)

        # If found significantly better opportunity, trigger full scan
        for cid, est_rpc in opportunities:
            if est_rpc > current_best_rpc * 1.3:  # 30% better
                logger.info(
                    f"Quick scan found opportunity: RPC ${est_rpc:.4f} vs current ${current_best_rpc:.4f}"
                )
                self._scan_and_select()
                return

    def _scan_and_select(self):
        """Scan markets and select best ones.

        Markets with existing positions are ALWAYS included (to unwind inventory).
        Remaining slots go to highest-reward markets.
        """
        self._last_scan = time.time()
        self._last_quick_scan = time.time()  # Reset quick scan timer too

        # Pass position CIDs to scanner so those markets bypass filters
        position_cids = {p["condition_id"] for p in self.db.get_all_positions()}
        results = self.scanner.scan(force_include_cids=position_cids)
        if not results:
            logger.warning("No viable markets found")
            return

        max_active = self.config.markets.max_active
        allocation = self.config.markets.allocation

        # Compute available capital (subtract existing positions)
        available_capital = self._compute_available_capital()
        if available_capital < 1:
            logger.warning(
                f"No available capital (${available_capital:.2f}) — "
                f"all funds in positions. Bot will only manage existing markets."
            )

        # Check if current markets should be kept
        current_ids = {s.market.condition_id for s in self.states}
        new_states: list[MarketState] = []

        # First pass: include any scanned market that has positions
        position_results = [r for r in results if r.market.condition_id in position_cids]
        non_position_results = [r for r in results if r.market.condition_id not in position_cids]

        # Combine: position markets first, then best remaining up to max_active
        combined = position_results + non_position_results
        if len(combined) > max_active:
            # Always keep position markets, trim the rest
            combined = position_results + non_position_results[:max(0, max_active - len(position_results))]

        if position_results:
            logger.info(
                f"Including {len(position_results)} markets with existing positions"
            )

        for i, result in enumerate(combined[:max_active]):
            if i >= len(allocation):
                break

            capital = available_capital * allocation[i] / 100
            cid = result.market.condition_id

            if cid in current_ids:
                # Keep existing state, just update metrics
                for s in self.states:
                    if s.market.condition_id == cid:
                        s.total_q_min = result.total_q_min
                        s.reward_per_q = result.reward_per_q
                        new_states.append(s)
                        break
            else:
                # New market
                token_yes = result.market.yes_token
                token_no = result.market.no_token

                # Load existing positions from DB (survives restart)
                saved_positions = []
                for p in self.db.get_positions(cid):
                    saved_positions.append(Position(
                        token_id=p["token_id"],
                        outcome=p["outcome"],
                        size=p["size"],
                        avg_cost=p["avg_cost"],
                    ))

                state = MarketState(
                    market=result.market,
                    allocated_capital=capital,
                    total_q_min=result.total_q_min,
                    reward_per_q=result.reward_per_q,
                    peak_value=0,  # will be set on first tick with real orderbook
                    positions=saved_positions,
                )

                # Persist
                if token_yes and token_no:
                    self.db.set_active_market(
                        cid, result.market.question,
                        token_yes.token_id, token_no.token_id,
                        capital, result.market.rewards.daily_rate,
                        result.market.rewards.max_spread,
                    )

                new_states.append(state)
                slog.market_enter(
                    market=result.market.question[:40],
                    condition_id=cid,
                    allocation=capital,
                    est_daily=result.estimated_daily_reward,
                )
                logger.info(
                    f"Selected market: {result.market.question[:50]} "
                    f"(${capital:.0f}, ${result.estimated_daily_reward:.2f}/day est)"
                )

        # Handle markets we're leaving
        leaving = current_ids - {s.market.condition_id for s in new_states}
        for s in self.states:
            if s.market.condition_id in leaving:
                # Check if we have positions and their P&L
                has_position = any(p.size > 0 for p in s.positions)

                if has_position and s.orderbook:
                    # Calculate position P&L
                    mid = s.orderbook.midpoint
                    position_pnl = 0.0
                    for pos in s.positions:
                        if pos.size > 0:
                            if pos.outcome == "Yes":
                                position_pnl += pos.size * (mid - pos.avg_cost)
                            else:  # No
                                position_pnl += pos.size * ((1 - mid) - pos.avg_cost)

                    position_value = sum(
                        p.size * (mid if p.outcome == "Yes" else 1 - mid)
                        for p in s.positions if p.size > 0
                    )
                    pnl_pct = position_pnl / position_value if position_value > 0 else 0

                    if pnl_pct < -0.01:  # Losing more than 1%
                        # Don't exit - enter unwind_only mode
                        logger.warning(
                            f"Position in {s.market.question[:30]} is losing {pnl_pct:.1%}, "
                            f"entering unwind-only mode"
                        )
                        s.unwind_only = True
                        s.unwind_start_time = time.time()
                        s.marked_for_exit = True
                        # Keep this market in states for unwinding
                        new_states.append(s)
                        slog.market_exit(
                            market=s.market.question[:40],
                            condition_id=s.market.condition_id,
                            reason="unwind_only_mode",
                        )
                        continue
                    else:
                        # Profitable or small loss - can exit
                        logger.info(
                            f"Position in {s.market.question[:30]} is {pnl_pct:+.1%}, "
                            f"closing and migrating"
                        )

                # No position or profitable - clean exit
                logger.info(f"Leaving market: {s.market.question[:40]}")
                slog.market_exit(
                    market=s.market.question[:40],
                    condition_id=s.market.condition_id,
                    reason="migration",
                )
                self.engine.emergency_cancel_all(s)
                self.db.remove_active_market(s.market.condition_id)

        self.states = new_states

    def _daily_reset(self, today: str):
        """Daily tasks: reset PnL, update CFR, record separated earnings."""
        self._last_daily_reset = today
        yesterday = self._last_daily_reset  # will be "today" for first reset

        # Aggregate separated P&L from all active markets
        total_spread = sum(s.pnl.spread_income for s in self.states)
        total_reward = sum(s.pnl.reward_income for s in self.states)
        total_gas = sum(s.pnl.gas_cost for s in self.states)
        total_adverse = sum(s.pnl.adverse_pnl for s in self.states)
        total_fills = sum(s.pnl.n_fills for s in self.states)
        n_markets = len(self.states)
        net = total_spread + total_reward - total_gas + total_adverse

        logger.info(f"=== Daily reset: {today} ===")
        logger.info(
            f"  Reward:  ${total_reward:.4f}  |  "
            f"Spread:  ${total_spread:.4f}  |  "
            f"Gas:  -${total_gas:.4f}  |  "
            f"Adverse: ${total_adverse:.4f}  |  "
            f"NET: ${net:.4f}  |  "
            f"Fills: {total_fills}"
        )

        # Record to strategy log
        slog.daily_pnl(
            date=today, reward=total_reward, fill_pnl=total_spread + total_adverse,
            gas=total_gas, net=net, n_fills=total_fills, n_markets=n_markets,
        )

        # Record to DB
        self.db.record_daily_pnl(
            date=today,
            starting_value=self.risk.daily_start_value,
            ending_value=self.risk.daily_start_value + net,
            spread_income=total_spread,
            reward_income=total_reward,
            realized_pnl=total_adverse,
            gas_cost=total_gas,
        )

        slog.daily_reset(date=today, portfolio_value=self.config.capital)

        # Reset per-market P&L counters
        for s in self.states:
            s.pnl.reset()

        # Update CFR with daily PnL as utility signal
        daily_pnl = self.risk.daily_pnl
        if daily_pnl != 0 and self.engine.cfr.strategies:
            strategies = self.engine.cfr.strategies
            # Estimate counterfactual: wider spread = less fill but safer,
            # narrower spread = more fills but riskier
            base = strategies[len(strategies) // 2]
            utilities = []
            for s in strategies:
                # Narrower spread → more volume but more adverse selection risk
                # Scale PnL by ratio: if we made money, narrower was better;
                # if we lost, wider was better
                ratio = base / s if s > 0 else 1.0
                if daily_pnl >= 0:
                    utilities.append(daily_pnl * ratio)
                else:
                    utilities.append(daily_pnl / ratio)
            self.engine.cfr.update(utilities)
            logger.info(f"CFR updated with daily PnL ${daily_pnl:.2f}, round {self.engine.cfr.rounds}")

        # Save CFR state
        self.engine.save_cfr_state()

        # Clean up old database records (keep 30 days)
        try:
            deleted = self.db.cleanup_old_data(days=30)
            if deleted > 0:
                logger.info(f"Database cleanup: removed {deleted} old records")
        except Exception as e:
            logger.warning(f"Database cleanup failed: {e}")

        # Reset risk manager daily counters
        self.risk.reset_daily(self.config.capital)

    def _log_status(self):
        """Log current status summary."""
        for state in self.states:
            ob = state.orderbook
            mid = ob.midpoint if ob else 0
            orders = len(state.active_orders)
            p = state.pnl
            logger.debug(
                f"[{state.market.question[:25]}] "
                f"mid={mid:.3f} orders={orders} "
                f"$/Q={state.reward_per_q:.4f} "
                f"rwd=${p.reward_income:.4f} spd=${p.spread_income:.4f} "
                f"gas=${p.gas_cost:.4f} fills={p.n_fills}"
            )

    def _shutdown(self):
        """Clean shutdown: cancel all orders, save state."""
        logger.info("Cancelling all orders...")
        for state in self.states:
            self.engine.emergency_cancel_all(state)

        self.engine.save_cfr_state()
        self.client.close()
        self.db.close()
        slog.bot_stop(reason="user")
        slog.close()
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Mantis - Polymarket Market Maker")
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without placing real orders",
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only scan markets and exit",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    if args.scan_only:
        client = PolymarketClient()
        scanner = MarketScanner(client, config)
        results = scanner.scan()
        print(f"\nTop {min(10, len(results))} markets for ${config.capital}:\n")
        for i, r in enumerate(results[:10]):
            print(
                f"  #{i+1}: ${r.estimated_daily_reward:.2f}/day "
                f"(${r.market.rewards.daily_rate}/d pool, "
                f"$/Q={r.reward_per_q:.4f}, "
                f"spread={r.spread_cents:.1f}c) "
                f"- {r.market.question[:60]}"
            )
        client.close()
        return

    # Run bot
    bot = Mantis(config, dry_run=args.dry_run)

    def handle_signal(sig, frame):
        bot.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    bot.start()


if __name__ == "__main__":
    main()
