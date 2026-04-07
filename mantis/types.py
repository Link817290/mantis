"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Token:
    token_id: str
    outcome: str  # "Yes" or "No"
    price: float


@dataclass
class MarketRewards:
    daily_rate: float  # USDC per day
    min_size: int
    max_spread: float  # in cents (e.g., 3.5)


@dataclass
class Market:
    condition_id: str
    question: str
    tokens: list[Token]
    rewards: MarketRewards
    end_date: str = ""
    active: bool = True

    @property
    def yes_token(self) -> Token | None:
        for t in self.tokens:
            if t.outcome == "Yes":
                return t
        return None

    @property
    def no_token(self) -> Token | None:
        for t in self.tokens:
            if t.outcome == "No":
                return t
        return None


@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class Orderbook:
    bids: list[OrderLevel]
    asks: list[OrderLevel]
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_cents(self) -> float:
        return (self.best_ask - self.best_bid) * 100


@dataclass
class ActiveOrder:
    order_id: str
    token_id: str
    side: Side
    price: float
    size: float
    market_condition_id: str = ""
    is_sell_yes: bool = False  # True if this is a SELL Yes order (recovers USDC)
    is_sell_no: bool = False   # True if this is a SELL No order (recovers USDC)


@dataclass
class Position:
    token_id: str
    outcome: str
    size: float  # number of shares held
    avg_cost: float  # average entry price


@dataclass
class PnlBreakdown:
    """Separated P&L tracking for a market."""
    spread_income: float = 0.0      # Realized spread capture from fills
    reward_income: float = 0.0      # Estimated reward accrual from Q share
    gas_cost: float = 0.0           # Gas fees for on-chain transactions
    adverse_pnl: float = 0.0        # Losses from adverse price moves after fills
    n_fills: int = 0                # Number of fills today
    n_buy_fills: int = 0
    n_sell_fills: int = 0
    total_buy_value: float = 0.0    # Sum of (price * size) for buy fills
    total_sell_value: float = 0.0   # Sum of (price * size) for sell fills

    def reset(self):
        self.spread_income = 0.0
        self.reward_income = 0.0
        self.gas_cost = 0.0
        self.adverse_pnl = 0.0
        self.n_fills = 0
        self.n_buy_fills = 0
        self.n_sell_fills = 0
        self.total_buy_value = 0.0
        self.total_sell_value = 0.0

    @property
    def net(self) -> float:
        return self.spread_income + self.reward_income - self.gas_cost + self.adverse_pnl


@dataclass
class MarketState:
    """Runtime state for a market we're actively making."""
    market: Market
    allocated_capital: float
    orderbook: Orderbook | None = None
    active_orders: list[ActiveOrder] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    recent_prices: list[float] = field(default_factory=list)
    recent_fills: list[dict] = field(default_factory=list)
    total_q_min: float = 0.0
    reward_per_q: float = 0.0
    peak_value: float = 0.0
    daily_pnl: float = 0.0
    pnl: PnlBreakdown = field(default_factory=PnlBreakdown)
    last_reprice: float = 0.0
    spread_multiplier: float = 1.0  # >1.0 when risk says WIDEN_SPREAD
    bid_cooldown_until: float = 0.0  # unix ts: don't place bids until this time
    ask_cooldown_until: float = 0.0  # unix ts: don't place asks until this time
    last_fill_time: float = 0.0
    volatility_3h: float = 0.0  # rolling 3h price volatility
    orders_cleaned_up: bool = False  # True after initial stale order cleanup
    # Migration/unwind mode
    unwind_only: bool = False  # True = only place orders to close positions, no new exposure
    unwind_start_time: float = 0.0  # When unwind mode started
    marked_for_exit: bool = False  # True = should exit when position closed
    # Real-time quality tracking
    quality_score: float = 1.0  # Current composite quality (0-1)
    activity_score: float = 1.0
    vpin_score: float = 1.0
    volatility_score: float = 1.0
    quality_trend: float = 0.0  # Positive = improving, negative = deteriorating
    last_quality_update: float = 0.0
    recent_trades: list = field(default_factory=list)  # For VPIN/vol calculation


@dataclass
class ScanResult:
    """Result from market scanner."""
    market: Market
    orderbook: Orderbook
    total_q_min: float
    your_q_min: float
    reward_per_q: float
    estimated_daily_reward: float
    spread_cents: float
    your_depth_share: float
    capital_needed: float = 0.0  # Capital locked for min_size on both sides
    reward_per_capital: float = 0.0  # est_daily / capital_needed
    # Quality metrics
    activity_score: float = 0.0
    fill_prob_score: float = 0.0
    volatility_score: float = 0.0
    vpin_score: float = 0.0
    price_score: float = 0.0
    quality_score: float = 0.0  # composite quality
    risk_adjusted_reward: float = 0.0  # reward_per_capital * quality_score
