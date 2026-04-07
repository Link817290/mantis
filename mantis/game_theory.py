"""Game theory algorithms for pricing decisions."""
from __future__ import annotations

import logging
import math
import time
import hashlib
import random
from collections import defaultdict, deque

from .types import Orderbook

logger = logging.getLogger("mantis.game_theory")


# ══════════════════════════════════════════════════════════════════════════════
# VPIN - 订单流毒性检测
# ══════════════════════════════════════════════════════════════════════════════

class VPINCalculator:
    """Volume-Synchronized Probability of Informed Trading.

    检测订单流是否有毒（信息交易者活跃）。
    VPIN 高时应该加宽价差或暂停报价。
    """

    def __init__(self, bucket_size: float = 30.0, n_buckets: int = 30):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.buckets: deque = deque(maxlen=n_buckets)
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0

    def update(self, size: float, price: float, mid: float):
        """每笔成交调用，使用 Bulk Volume Classification 推断方向"""
        # BVC: 成交价 vs mid 推断买卖方向
        if mid > 0:
            deviation = (price - mid) / mid
            buy_pct = 0.5 + min(0.5, max(-0.5, deviation * 50))
        else:
            buy_pct = 0.5

        self.current_bucket_buy += size * buy_pct
        self.current_bucket_volume += size

        # Bucket 满了，计算 imbalance
        while self.current_bucket_volume >= self.bucket_size:
            overflow = self.current_bucket_volume - self.bucket_size
            ratio = (self.bucket_size - overflow) / self.current_bucket_volume if self.current_bucket_volume > 0 else 1

            bucket_buy = self.current_bucket_buy * ratio
            bucket_sell = (self.bucket_size - overflow) - bucket_buy
            imbalance = abs(bucket_buy - bucket_sell) / self.bucket_size if self.bucket_size > 0 else 0
            self.buckets.append(imbalance)

            self.current_bucket_buy = self.current_bucket_buy * (1 - ratio)
            self.current_bucket_volume = overflow

    def get_vpin(self) -> float:
        """返回 VPIN [0,1]，越高越危险"""
        if len(self.buckets) < 5:
            return 0.15
        return sum(self.buckets) / len(self.buckets)

    def is_toxic(self, threshold: float = 0.35) -> bool:
        return self.get_vpin() > threshold

    def reset(self):
        self.buckets.clear()
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 价格冲击检测 - 防止被吃单
# ══════════════════════════════════════════════════════════════════════════════

class EatDetector:
    """检测即将被吃单的信号，提前撤单"""

    def __init__(self):
        self.price_history: deque = deque(maxlen=60)  # 最近 60 个 tick
        self.depth_history: deque = deque(maxlen=20)
        self.last_alert_time = 0.0

    def update(self, mid: float, bid_depth: float, ask_depth: float):
        """每个 tick 更新"""
        now = time.time()
        self.price_history.append((now, mid))
        self.depth_history.append((now, bid_depth, ask_depth))

    def detect_momentum(self, lookback_sec: float = 30) -> tuple[float, str]:
        """检测价格动量。

        Returns: (momentum, direction)
            momentum: 动量强度 [0, 1]
            direction: "up", "down", "neutral"
        """
        if len(self.price_history) < 5:
            return 0.0, "neutral"

        now = time.time()
        recent = [(t, p) for t, p in self.price_history if now - t <= lookback_sec]

        if len(recent) < 3:
            return 0.0, "neutral"

        # 计算价格变化
        first_price = recent[0][1]
        last_price = recent[-1][1]
        change = last_price - first_price

        # 计算变化速度
        time_span = recent[-1][0] - recent[0][0]
        if time_span < 1:
            return 0.0, "neutral"

        velocity = abs(change) / time_span * 60  # 每分钟变化

        # 归一化到 [0, 1]，假设 5% 每分钟是极端情况
        momentum = min(1.0, velocity / 0.05)

        if change > 0.005:
            direction = "up"
        elif change < -0.005:
            direction = "down"
        else:
            direction = "neutral"

        return momentum, direction

    def detect_depth_drain(self, side: str) -> bool:
        """检测某一侧深度是否在快速消耗（有人在扫单）"""
        if len(self.depth_history) < 5:
            return False

        recent = list(self.depth_history)[-10:]

        if side == "bid":
            depths = [d[1] for d in recent]
        else:
            depths = [d[2] for d in recent]

        if len(depths) < 3 or depths[0] == 0:
            return False

        # 检查深度是否在持续减少
        decreasing_count = sum(1 for i in range(1, len(depths)) if depths[i] < depths[i-1])
        drain_ratio = depths[-1] / depths[0] if depths[0] > 0 else 1

        # 超过 60% 的 tick 深度在减少，且总深度减少超过 30%
        return decreasing_count > len(depths) * 0.6 and drain_ratio < 0.7

    def should_retreat(self, my_side: str, my_price: float, mid: float) -> tuple[bool, str]:
        """判断是否应该撤单。

        Args:
            my_side: "bid" or "ask"
            my_price: 我的报价
            mid: 当前中间价

        Returns: (should_retreat, reason)
        """
        momentum, direction = self.detect_momentum()

        # 情况1: 强动量朝我的订单方向来
        if my_side == "bid" and direction == "down" and momentum > 0.5:
            return True, f"price_falling_fast(m={momentum:.2f})"
        if my_side == "ask" and direction == "up" and momentum > 0.5:
            return True, f"price_rising_fast(m={momentum:.2f})"

        # 情况2: 我这一侧的深度在被消耗
        if self.detect_depth_drain(my_side):
            return True, "depth_draining"

        # 情况3: 价格已经非常接近我的订单
        distance = abs(my_price - mid)
        if distance < 0.003:  # 不到 0.3 分
            return True, "price_too_close"

        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# 随机化 - 防止策略被预测
# ══════════════════════════════════════════════════════════════════════════════

class Randomizer:
    """给报价添加随机性，防止被狙击"""

    def __init__(self, seed: str = ""):
        self.seed = seed or str(time.time())

    def jitter_price(self, price: float, max_jitter_cents: float = 0.2) -> float:
        """给价格添加小幅随机抖动"""
        # 每 10 分钟换一次随机种子
        period_seed = f"{self.seed}:{int(time.time() // 600)}"
        rng = random.Random(hashlib.md5(period_seed.encode()).hexdigest())
        jitter = (rng.random() - 0.5) * 2 * (max_jitter_cents / 100)
        return round(price + jitter, 3)

    def jitter_size(self, size: float, max_jitter_pct: float = 0.1) -> int:
        """给订单大小添加随机抖动"""
        period_seed = f"{self.seed}:size:{int(time.time() // 600)}"
        rng = random.Random(hashlib.md5(period_seed.encode()).hexdigest())
        multiplier = 1 + (rng.random() - 0.5) * 2 * max_jitter_pct
        return max(1, int(size * multiplier))

    def jitter_interval(self, base_sec: float) -> float:
        """随机化刷新间隔"""
        return base_sec * random.uniform(0.85, 1.15)


# ── Markov Price Drift ──

def markov_price_drift(recent_prices: list[float], bins: int = 5) -> float:
    """Estimate price drift using discrete Markov transition matrix.

    Discretizes prices into bins, builds transition matrix,
    returns expected drift from current state.
    """
    if len(recent_prices) < 3:
        return 0.0

    # Discretize into bins
    min_p = min(recent_prices)
    max_p = max(recent_prices)
    rng = max_p - min_p
    if rng < 0.001:
        return 0.0  # price is flat

    bin_size = rng / bins

    def to_bin(p: float) -> int:
        b = int((p - min_p) / bin_size)
        return min(b, bins - 1)

    def bin_center(b: int) -> float:
        return min_p + (b + 0.5) * bin_size

    # Build transition counts
    transitions: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(len(recent_prices) - 1):
        s_from = to_bin(recent_prices[i])
        s_to = to_bin(recent_prices[i + 1])
        transitions[s_from][s_to] += 1

    # Current state
    current_bin = to_bin(recent_prices[-1])
    current_center = bin_center(current_bin)

    # Expected next price
    row = transitions.get(current_bin, {})
    total = sum(row.values())
    if total == 0:
        return 0.0

    expected = sum(bin_center(b) * count / total for b, count in row.items())
    drift = expected - current_center

    return drift


# ── Glosten-Milgrom Minimum Spread ──

def glosten_milgrom_min_spread(
    recent_fills: list[dict],
    informed_ratio: float = 0.15,
    expected_jump: float = 0.05,
    vpin: float | None = None,
) -> float:
    """Compute minimum safe spread using Glosten-Milgrom model.

    min_spread >= mu * expected_instant_price_jump

    Where mu = estimated fraction of informed traders.

    改进：
    1. 如果提供了 VPIN，直接用 VPIN 作为 mu 的估计
    2. 动态计算 expected_jump 基于近期价格波动
    """
    # 如果有 VPIN，优先使用
    if vpin is not None:
        adjusted_mu = max(0.1, min(0.6, vpin * 1.2))
    else:
        if not recent_fills:
            return informed_ratio * expected_jump

        # 检测逆向选择：大单 + 单边集中
        total_size = sum(f.get("size", 0) for f in recent_fills)
        avg_size = total_size / len(recent_fills) if recent_fills else 0

        adverse_count = 0
        buy_count = 0
        for fill in recent_fills:
            size = fill.get("size", 0)
            if size > avg_size * 1.5:  # 动态阈值：比平均大 50%
                adverse_count += 1
            if fill.get("side") == "BUY":
                buy_count += 1

        # 单边集中度
        one_side_ratio = max(buy_count, len(recent_fills) - buy_count) / len(recent_fills) if recent_fills else 0.5

        adverse_rate = adverse_count / len(recent_fills) if recent_fills else 0
        adjusted_mu = min(informed_ratio + adverse_rate * 0.3 + (one_side_ratio - 0.5) * 0.2, 0.5)

    # 动态计算预期跳跃
    if recent_fills and len(recent_fills) >= 2:
        prices = [f.get("price", 0.5) for f in recent_fills[-20:]]
        if len(prices) >= 2:
            jumps = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            avg_jump = sum(jumps) / len(jumps) if jumps else expected_jump
            expected_jump = max(expected_jump, avg_jump * 1.5)

    min_spread = adjusted_mu * expected_jump
    return min_spread


# ── Nash Optimal Spread ──

def nash_optimal_spread(
    orderbook: Orderbook,
    min_spread: float,
    aggression: float = 0.9,
) -> float:
    """Determine spread based on competitor analysis (Nash/Bertrand).

    Look at existing orderbook to see what competitors offer.
    Set our spread slightly tighter (aggression < 1) but never below min_spread.
    """
    if not orderbook.bids or not orderbook.asks:
        return max(min_spread, 0.01)

    # Competitor spread = current best bid-ask spread
    competitor_spread = orderbook.best_ask - orderbook.best_bid

    # We want to be slightly tighter than competitors
    our_spread = competitor_spread * aggression

    # But never below the GM safety floor
    our_spread = max(our_spread, min_spread)

    # And never below 0.2 cents (gas costs make sub-cent impractical)
    our_spread = max(our_spread, 0.002)

    return our_spread


# ── CFR (Counterfactual Regret Minimization) ──

class CFREngine:
    """Simple CFR for spread strategy selection.

    Strategies: different half-spread values.
    Utility: daily PnL from using that spread.
    """

    def __init__(self, strategies: list[float]):
        self.strategies = strategies
        self.n = len(strategies)
        self.cumulative_regret = [0.0] * self.n
        self.cumulative_strategy = [0.0] * self.n
        self.rounds = 0

    def get_strategy(self) -> list[float]:
        """Get current mixed strategy (probability distribution over spreads)."""
        # Regret-matching: positive regrets determine probabilities
        positive_regret = [max(0, r) for r in self.cumulative_regret]
        total = sum(positive_regret)

        if total > 0:
            return [r / total for r in positive_regret]
        else:
            # Uniform if no positive regret
            return [1.0 / self.n] * self.n

    def select_spread(self) -> float:
        """Select a half-spread based on current strategy distribution."""
        import random
        probs = self.get_strategy()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return self.strategies[i]
        return self.strategies[-1]

    def update(self, utilities: list[float]):
        """Update regrets given utility (PnL) for each strategy.

        utilities[i] = what we would have earned using strategy i.
        """
        strategy = self.get_strategy()
        expected_utility = sum(s * u for s, u in zip(strategy, utilities))

        for i in range(self.n):
            regret = utilities[i] - expected_utility
            self.cumulative_regret[i] += regret
            self.cumulative_strategy[i] += strategy[i]

        self.rounds += 1

    def get_average_strategy(self) -> list[float]:
        """Get the time-averaged strategy (converges to Nash equilibrium)."""
        total = sum(self.cumulative_strategy)
        if total > 0:
            return [s / total for s in self.cumulative_strategy]
        return [1.0 / self.n] * self.n

    def load_state(self, state: list[dict]):
        """Load CFR state from database."""
        for row in state:
            key = row["strategy_key"]
            try:
                idx = self.strategies.index(float(key))
                self.cumulative_regret[idx] = row["cumulative_regret"]
                self.cumulative_strategy[idx] = row["cumulative_strategy"]
                self.rounds = max(self.rounds, row["rounds"])
            except (ValueError, IndexError):
                continue

    def save_state(self) -> list[tuple[str, float, float, int]]:
        """Return state tuples for database persistence."""
        return [
            (str(s), self.cumulative_regret[i], self.cumulative_strategy[i], self.rounds)
            for i, s in enumerate(self.strategies)
        ]


# ── Evolutionary Game Theory: Market Ecosystem Detection ──

def detect_ecosystem_state(
    orderbook: Orderbook,
    prev_orderbook: Orderbook | None,
) -> str:
    """Detect market ecosystem state changes.

    Returns: 'normal', 'crowded', 'dead', 'whale_entered'
    """
    if not orderbook.bids or not orderbook.asks:
        return "dead"

    current_depth = sum(l.size for l in orderbook.bids) + sum(l.size for l in orderbook.asks)

    if current_depth < 100:
        return "dead"

    if prev_orderbook and prev_orderbook.bids and prev_orderbook.asks:
        prev_depth = sum(l.size for l in prev_orderbook.bids) + sum(l.size for l in prev_orderbook.asks)

        if prev_depth > 0:
            ratio = current_depth / prev_depth
            if ratio > 3.0:
                return "whale_entered"

    # Check if spread is too tight (price war)
    if orderbook.spread_cents < 0.3:
        return "crowded"

    return "normal"


# ══════════════════════════════════════════════════════════════════════════════
# 最优平仓策略 - 被吃单后的最优解
# ══════════════════════════════════════════════════════════════════════════════

class UnwindOptimizer:
    """被吃单后的最优平仓策略"""

    def __init__(
        self,
        min_profit_pct: float = 0.005,   # 最小期望利润 0.5%
        max_loss_pct: float = 0.03,       # 最大可接受亏损 3%
        time_decay_hours: float = 4.0,    # 时间衰减：4小时后开始降低要求
        urgent_loss_pct: float = 0.05,    # 紧急止损线 5%
    ):
        self.min_profit_pct = min_profit_pct
        self.max_loss_pct = max_loss_pct
        self.time_decay_hours = time_decay_hours
        self.urgent_loss_pct = urgent_loss_pct

    def compute_unwind_price(
        self,
        outcome: str,          # "Yes" or "No"
        avg_cost: float,       # 持仓成本
        current_mid: float,    # 当前中间价
        hours_held: float,     # 持仓时间（小时）
        orderbook: Orderbook,  # 当前订单簿
    ) -> tuple[float, str, int]:
        """计算最优平仓价格。

        Returns: (unwind_price, strategy, urgency)
            strategy: "profit", "breakeven", "cut_loss", "urgent_stop"
            urgency: 0=正常, 1=尽快, 2=立即
        """
        if outcome == "Yes":
            current_price = current_mid
            best_bid = orderbook.bids[0].price if orderbook.bids else current_mid * 0.98
        else:
            current_price = 1 - current_mid
            # NO 的 best_bid 是 YES 的 best_ask 的补数
            best_bid = 1 - orderbook.asks[0].price if orderbook.asks else current_price * 0.98

        # 计算当前盈亏
        pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0

        # 时间衰减系数：持仓越久，对利润的要求越低
        time_factor = max(0, 1 - hours_held / self.time_decay_hours)
        adjusted_min_profit = self.min_profit_pct * time_factor

        # 策略1: 紧急止损 - 亏损超过阈值，立即市价平仓
        if pnl_pct < -self.urgent_loss_pct:
            return best_bid, "urgent_stop", 2

        # 策略2: 止损 - 亏损超过最大容忍，挂单平仓
        if pnl_pct < -self.max_loss_pct:
            # 挂在 best_bid 上方一点，尽快成交
            unwind_price = round(best_bid * 1.002, 3)
            return unwind_price, "cut_loss", 1

        # 策略3: 保本 - 小亏或持仓时间长
        if pnl_pct < adjusted_min_profit or hours_held > self.time_decay_hours:
            # 尝试以成本价平仓
            unwind_price = round(avg_cost * 1.002, 3)  # 加一点点利润
            return unwind_price, "breakeven", 0

        # 策略4: 盈利平仓 - 有利润，挂单等待
        target_profit = max(adjusted_min_profit, self.min_profit_pct)
        unwind_price = round(avg_cost * (1 + target_profit), 3)

        # 但不能高于当前价格太多，否则永远成交不了
        max_reasonable = current_price * 1.02
        unwind_price = min(unwind_price, max_reasonable)

        return unwind_price, "profit", 0

    def should_market_sell(
        self,
        pnl_pct: float,
        hours_held: float,
        momentum: float,
        momentum_direction: str,
        my_side: str,  # "Yes" or "No"
    ) -> bool:
        """判断是否应该市价卖出（接受亏损）。

        当价格快速朝不利方向移动时，应该立即平仓。
        """
        # 亏损 + 价格继续朝不利方向移动
        if pnl_pct < -0.01:  # 已经亏损 1%
            if my_side == "Yes" and momentum_direction == "down" and momentum > 0.4:
                return True
            if my_side == "No" and momentum_direction == "up" and momentum > 0.4:
                return True

        # 持仓时间过长 + 亏损
        if hours_held > 8 and pnl_pct < -0.02:
            return True

        return False

    def get_ladder_prices(
        self,
        avg_cost: float,
        current_mid: float,
        total_size: float,
        n_levels: int = 3,
    ) -> list[tuple[float, float]]:
        """生成阶梯平仓价格（分批平仓）。

        Returns: [(price1, size1), (price2, size2), ...]
        """
        if n_levels < 1:
            return [(avg_cost * 1.01, total_size)]

        prices = []
        size_per_level = total_size / n_levels

        for i in range(n_levels):
            # 第一档最接近当前价，后面的档位利润更高
            profit_target = 0.005 + i * 0.005  # 0.5%, 1%, 1.5%
            price = round(avg_cost * (1 + profit_target), 3)

            # 不能超过合理范围
            if price > 0.99:
                price = 0.99

            prices.append((price, size_per_level))

        return prices
