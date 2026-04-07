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

LOG_DIR = Path("/workspace/mantis/data")
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


# Module-level singleton
slog = StrategyLogger()
