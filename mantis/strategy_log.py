"""Structured strategy logging for post-hoc analysis and tuning.

Writes JSON-lines to a rotating log file. Each line is a self-contained event
that can be loaded into pandas or piped through jq for analysis.

Usage in other modules:
    from .strategy_log import slog
    slog.order_placed(market="NYC Temp", side="BUY", price=0.42, ...)
    slog.fill(market="NYC Temp", side="BUY", fill_price=0.42, ...)

Analysis:
    cat data/strategy.jsonl | jq 'select(.event=="fill")'
    cat data/strategy.jsonl | jq 'select(.event=="scan_result")' | jq -s 'sort_by(.rpc) | reverse'
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

_MODULE_DIR = Path(__file__).parent.parent.resolve()
LOG_DIR = _MODULE_DIR / "data"
LOG_FILE = LOG_DIR / "strategy.jsonl"

_logger = logging.getLogger("mantis.strategy_log")


class StrategyLogger:
    """Append-only JSON-lines logger for strategy events."""

    def __init__(self, path: Path = LOG_FILE):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self):
        if self._file is None or self._file.closed:
            self._file = open(self._path, "a")

    def _write(self, event: str, **fields):
        self._ensure_open()
        record = {"ts": time.time(), "event": event, **fields}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()

    # ── Scan Events ──

    def scan_start(self, n_candidates: int):
        self._write("scan_start", n_candidates=n_candidates)

    def scan_result(self, market: str, condition_id: str, daily_rate: float,
                    total_q: float, your_q: float, est_daily: float,
                    rpc: float, spread_cents: float, cap_needed: float):
        self._write("scan_result", market=market, condition_id=condition_id,
                    daily_rate=daily_rate, total_q=total_q, your_q=your_q,
                    est_daily=est_daily, rpc=rpc, spread_cents=spread_cents,
                    cap_needed=cap_needed)

    def scan_skip(self, market: str, reason: str, **details):
        self._write("scan_skip", market=market, reason=reason, **details)

    def market_selected(self, rank: int, market: str, condition_id: str,
                        allocation_pct: float):
        self._write("market_selected", rank=rank, market=market,
                    condition_id=condition_id, allocation_pct=allocation_pct)

    # ── Order Events ──

    def order_placed(self, market: str, side: str, price: float, size: float,
                     mid: float, half_spread: float, spread_pct: float,
                     q_score: float = 0):
        self._write("order_placed", market=market, side=side, price=price,
                    size=size, mid=mid, half_spread=half_spread,
                    spread_pct=spread_pct, q_score=q_score)

    def order_cancelled(self, market: str, side: str, reason: str):
        self._write("order_cancelled", market=market, side=side, reason=reason)

    def reprice(self, market: str, old_bid: float, old_ask: float,
                new_bid: float, new_ask: float, mid: float,
                trigger: str = "drift"):
        self._write("reprice", market=market, old_bid=old_bid, old_ask=old_ask,
                    new_bid=new_bid, new_ask=new_ask, mid=mid, trigger=trigger)

    # ── Fill Events ──

    def fill(self, market: str, side: str, price: float, size: float,
             mid_at_fill: float, spread_at_fill: float = 0,
             queue_depth: float = 0, inventory_after: float = 0):
        self._write("fill", market=market, side=side, price=price, size=size,
                    mid_at_fill=mid_at_fill, spread_at_fill=spread_at_fill,
                    queue_depth=queue_depth, inventory_after=inventory_after)

    def fill_adverse_check(self, market: str, side: str, price: float,
                           mid_at_fill: float, mid_after_60s: float,
                           is_adverse: bool, move_bps: float):
        self._write("fill_adverse_check", market=market, side=side,
                    price=price, mid_at_fill=mid_at_fill,
                    mid_after_60s=mid_after_60s, is_adverse=is_adverse,
                    move_bps=move_bps)

    # ── Risk Events ──

    def risk_trigger(self, market: str, rule: str, action: str, **details):
        self._write("risk_trigger", market=market, rule=rule, action=action,
                    **details)

    def cooldown_start(self, market: str, side: str, duration_sec: int):
        self._write("cooldown_start", market=market, side=side,
                    duration_sec=duration_sec)

    def inventory_update(self, market: str, yes_qty: float, no_qty: float,
                         yes_avg_cost: float, unrealized_pnl: float):
        self._write("inventory_update", market=market, yes_qty=yes_qty,
                    no_qty=no_qty, yes_avg_cost=yes_avg_cost,
                    unrealized_pnl=unrealized_pnl)

    # ── P&L Events ──

    def daily_pnl(self, date: str, reward: float, fill_pnl: float,
                  gas: float, net: float, n_fills: int, n_markets: int):
        self._write("daily_pnl", date=date, reward=reward, fill_pnl=fill_pnl,
                    gas=gas, net=net, n_fills=n_fills, n_markets=n_markets)

    def take_profit(self, market: str, qty: float, entry_price: float,
                    exit_price: float, pnl: float):
        self._write("take_profit", market=market, qty=qty,
                    entry_price=entry_price, exit_price=exit_price, pnl=pnl)

    # ── API Operation Events ──

    def api_call(self, method: str, endpoint: str, status: str = "ok",
                 latency_ms: float = 0, **details):
        self._write("api_call", method=method, endpoint=endpoint,
                    status=status, latency_ms=latency_ms, **details)

    def api_order_submit(self, market: str, side: str, price: float,
                         size: float, order_id: str = "", dry_run: bool = False):
        self._write("api_order_submit", market=market, side=side, price=price,
                    size=size, order_id=order_id, dry_run=dry_run)

    def api_order_cancel(self, market: str, order_id: str, reason: str = "reprice"):
        self._write("api_order_cancel", market=market, order_id=order_id,
                    reason=reason)

    def api_error(self, method: str, endpoint: str, error: str, **details):
        self._write("api_error", method=method, endpoint=endpoint,
                    error=error, **details)

    # ── Lifecycle Events ──

    def bot_start(self, capital: float, mode: str, max_markets: int):
        self._write("bot_start", capital=capital, mode=mode,
                    max_markets=max_markets)

    def bot_stop(self, reason: str = "user"):
        self._write("bot_stop", reason=reason)

    def market_enter(self, market: str, condition_id: str,
                     allocation: float, est_daily: float):
        self._write("market_enter", market=market, condition_id=condition_id,
                    allocation=allocation, est_daily=est_daily)

    def market_exit(self, market: str, condition_id: str, reason: str):
        self._write("market_exit", market=market, condition_id=condition_id,
                    reason=reason)

    def daily_reset(self, date: str, portfolio_value: float):
        self._write("daily_reset", date=date, portfolio_value=portfolio_value)

    # ── System Events ──

    def tick(self, market: str, mid: float, spread_cents: float,
             vol_3h: float, bid_live: bool, ask_live: bool,
             yes_inventory: float = 0, spread_multiplier: float = 1.0):
        self._write("tick", market=market, mid=mid, spread_cents=spread_cents,
                    vol_3h=vol_3h, bid_live=bid_live, ask_live=ask_live,
                    yes_inventory=yes_inventory,
                    spread_multiplier=spread_multiplier)

    def error(self, context: str, message: str, **details):
        self._write("error", context=context, message=message, **details)

    # ── 防吃单事件 ──

    def retreat(self, market: str, side: str, price: float, reason: str):
        """记录撤单事件（检测到被吃风险）"""
        self._write("retreat", market=market, side=side, price=price, reason=reason)

    def stop_loss(self, market: str, outcome: str, size: float, avg_cost: float,
                  sell_price: float, pnl_pct: float, reason: str):
        """记录止损事件"""
        self._write("stop_loss", market=market, outcome=outcome, size=size,
                    avg_cost=avg_cost, sell_price=sell_price, pnl_pct=pnl_pct,
                    reason=reason)

    def vpin_alert(self, market: str, vpin: float, action: str):
        """记录 VPIN 警报"""
        self._write("vpin_alert", market=market, vpin=vpin, action=action)

    def momentum_alert(self, market: str, momentum: float, direction: str,
                       action: str):
        """记录动量警报"""
        self._write("momentum_alert", market=market, momentum=momentum,
                    direction=direction, action=action)

    def quality_alert(self, market: str, alert: str, quality: float,
                      vpin: float, activity: float):
        """记录质量警报"""
        self._write("quality_alert", market=market, alert=alert,
                    quality=quality, vpin=vpin, activity=activity)

    # ── 滑点追踪事件 ──

    def fill_slippage(self, market: str, side: str, expected_price: float,
                      actual_price: float, slippage_cents: float, size: float,
                      orderbook_age_sec: float = 0):
        """记录成交滑点"""
        self._write("fill_slippage", market=market, side=side,
                    expected_price=expected_price, actual_price=actual_price,
                    slippage_cents=slippage_cents, size=size,
                    orderbook_age_sec=orderbook_age_sec)

    def slippage_alert(self, market: str, side: str, slippage_cents: float,
                       max_allowed: float, action: str):
        """记录滑点警报（超过阈值）"""
        self._write("slippage_alert", market=market, side=side,
                    slippage_cents=slippage_cents, max_allowed=max_allowed,
                    action=action)

    def orderbook_stale(self, market: str, age_sec: float, max_age: float):
        """记录订单簿过期"""
        self._write("orderbook_stale", market=market, age_sec=age_sec,
                    max_age=max_age)

    # ── 模型迭代数据 ──

    def orderbook_snapshot(self, market: str, mid: float, spread_cents: float,
                           bid_depth_5c: float, ask_depth_5c: float,
                           imbalance: float, n_bid_levels: int, n_ask_levels: int,
                           best_bid: float, best_ask: float,
                           bid_depth_10c: float = 0, ask_depth_10c: float = 0):
        """下单时的订单簿快照 - 用于分析最优spread"""
        self._write("orderbook_snapshot", market=market, mid=mid,
                    spread_cents=spread_cents, bid_depth_5c=bid_depth_5c,
                    ask_depth_5c=ask_depth_5c, imbalance=imbalance,
                    n_bid_levels=n_bid_levels, n_ask_levels=n_ask_levels,
                    best_bid=best_bid, best_ask=best_ask,
                    bid_depth_10c=bid_depth_10c, ask_depth_10c=ask_depth_10c)

    def order_lifecycle(self, market: str, order_id: str, side: str,
                        price: float, size: float, event: str,
                        age_sec: float = 0, queue_position: float = 0,
                        queue_ahead: float = 0):
        """订单生命周期事件 - 用于分析fill probability"""
        # event: "placed", "partial_fill", "full_fill", "cancelled", "expired"
        self._write("order_lifecycle", market=market, order_id=order_id,
                    side=side, price=price, size=size, event=event,
                    age_sec=age_sec, queue_position=queue_position,
                    queue_ahead=queue_ahead)

    def queue_position(self, market: str, side: str, price: float,
                       your_size: float, queue_ahead: float, total_at_price: float,
                       est_fill_prob: float):
        """队列位置追踪 - 用于分析竞争环境"""
        self._write("queue_position", market=market, side=side, price=price,
                    your_size=your_size, queue_ahead=queue_ahead,
                    total_at_price=total_at_price, est_fill_prob=est_fill_prob)

    def competition_snapshot(self, market: str, n_makers: int,
                             tightest_spread: float, avg_spread: float,
                             total_bid_depth: float, total_ask_depth: float,
                             your_q_share: float):
        """竞争环境快照 - 用于分析市场拥挤度"""
        self._write("competition_snapshot", market=market, n_makers=n_makers,
                    tightest_spread=tightest_spread, avg_spread=avg_spread,
                    total_bid_depth=total_bid_depth, total_ask_depth=total_ask_depth,
                    your_q_share=your_q_share)

    def price_trajectory(self, market: str, fill_id: str, fill_side: str,
                         fill_price: float, mid_at_fill: float,
                         mid_after_10s: float, mid_after_30s: float,
                         mid_after_60s: float, mid_after_300s: float,
                         max_adverse_move: float, realized_pnl: float):
        """成交后价格轨迹 - 用于分析adverse selection"""
        self._write("price_trajectory", market=market, fill_id=fill_id,
                    fill_side=fill_side, fill_price=fill_price,
                    mid_at_fill=mid_at_fill, mid_after_10s=mid_after_10s,
                    mid_after_30s=mid_after_30s, mid_after_60s=mid_after_60s,
                    mid_after_300s=mid_after_300s, max_adverse_move=max_adverse_move,
                    realized_pnl=realized_pnl)

    def spread_decision(self, market: str, chosen_spread: float,
                        min_spread: float, vol_adjusted: float,
                        inventory_adjusted: float, vpin_adjusted: float,
                        final_spread: float, spread_vs_market: float,
                        expected_q_score: float):
        """Spread决策记录 - 用于优化spread策略"""
        self._write("spread_decision", market=market, chosen_spread=chosen_spread,
                    min_spread=min_spread, vol_adjusted=vol_adjusted,
                    inventory_adjusted=inventory_adjusted, vpin_adjusted=vpin_adjusted,
                    final_spread=final_spread, spread_vs_market=spread_vs_market,
                    expected_q_score=expected_q_score)

    def q_score_update(self, market: str, your_q: float, total_q: float,
                       q_share_pct: float, reward_rate: float,
                       est_hourly_reward: float, q_rank: int = 0):
        """Q值更新 - 用于分析reward效率"""
        self._write("q_score_update", market=market, your_q=your_q,
                    total_q=total_q, q_share_pct=q_share_pct,
                    reward_rate=reward_rate, est_hourly_reward=est_hourly_reward,
                    q_rank=q_rank)

    def trade_flow(self, market: str, window_sec: int, buy_volume: float,
                   sell_volume: float, net_flow: float, vpin: float,
                   n_trades: int, avg_trade_size: float,
                   large_trade_pct: float):
        """交易流分析 - 用于toxicity检测"""
        self._write("trade_flow", market=market, window_sec=window_sec,
                    buy_volume=buy_volume, sell_volume=sell_volume,
                    net_flow=net_flow, vpin=vpin, n_trades=n_trades,
                    avg_trade_size=avg_trade_size, large_trade_pct=large_trade_pct)

    def model_feature(self, market: str, feature_set: str, **features):
        """通用特征记录 - 用于ML模型训练"""
        self._write("model_feature", market=market, feature_set=feature_set,
                    **features)


# Module-level singleton
slog = StrategyLogger()
