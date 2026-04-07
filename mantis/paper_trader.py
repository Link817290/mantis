"""Paper trading module - track virtual orders against real market data.

Connects to Polymarket APIs, places virtual orders (no real money),
monitors real trade flow to determine if orders would have been filled.
Records all statistics for calibrating backtest parameters.

Usage:
    python3 -m mantis.paper_trader              # run with defaults
    python3 -m mantis.paper_trader --top 5      # track top 5 markets
    python3 -m mantis.paper_trader --stats       # show collected stats
    python3 -m mantis.paper_trader --calibrate   # output backtest parameters
"""
from __future__ import annotations

import json
import logging
import math
import signal
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, MantisConfig
from .polymarket_client import PolymarketClient
from .scanner import compute_total_q_min, estimate_your_q

DATA_API_BASE = "https://data-api.polymarket.com"

logger = logging.getLogger("mantis.paper")

# Cross-platform path: use the directory where this module is located
_MODULE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = _MODULE_DIR / "data" / "paper_trades.db"


# ── Data types ──

@dataclass
class VirtualOrder:
    """A simulated order we're tracking."""
    order_id: str
    condition_id: str
    token_id: str
    side: str           # "BUY" or "SELL"
    price: float
    size: float
    placed_at: float    # unix timestamp
    mid_at_place: float # midpoint when placed
    status: str = "live"  # live / filled / cancelled
    filled_at: float = 0.0
    fill_price: float = 0.0
    mid_at_fill: float = 0.0
    mid_after_60s: float = 0.0   # price 60s after fill (adverse selection check)
    queue_depth_at_place: float = 0.0  # estimated queue ahead of us
    # === 新增: 时间优先队列追踪 ===
    queue_position: float = 0.0       # 我们在队列中的位置(前面有多少量)
    volume_filled_at_level: float = 0.0  # 该价位已成交的总量(用于追踪队列推进)
    join_sequence: int = 0            # 加入队列的顺序号


@dataclass
class InformedTraderStats:
    """Tracks patterns that indicate informed/toxic order flow."""
    recent_trades: list = field(default_factory=list)  # (ts, price, size, side)
    vpin_buckets: list = field(default_factory=list)   # (buy_vol, sell_vol) per bucket
    bucket_volume: float = 0.0
    current_bucket_buy: float = 0.0
    current_bucket_sell: float = 0.0

    def add_trade(self, ts: float, price: float, size: float, side: str, mid: float):
        """添加一笔交易并更新统计"""
        self.recent_trades.append((ts, price, size, side, mid))
        # 保留最近200笔
        if len(self.recent_trades) > 200:
            self.recent_trades = self.recent_trades[-200:]

        # VPIN bucket (每30单位量一个bucket)
        if side == "BUY":
            self.current_bucket_buy += size
        else:
            self.current_bucket_sell += size
        self.bucket_volume += size

        if self.bucket_volume >= 30:
            self.vpin_buckets.append((self.current_bucket_buy, self.current_bucket_sell))
            if len(self.vpin_buckets) > 30:
                self.vpin_buckets = self.vpin_buckets[-30:]
            self.bucket_volume = 0
            self.current_bucket_buy = 0
            self.current_bucket_sell = 0

    def get_vpin(self) -> float:
        """计算当前VPIN值 (0-1, 越高越toxic)"""
        if len(self.vpin_buckets) < 5:
            return 0.0
        total_imbalance = sum(abs(b - s) for b, s in self.vpin_buckets)
        total_volume = sum(b + s for b, s in self.vpin_buckets)
        return total_imbalance / total_volume if total_volume > 0 else 0.0

    def get_momentum(self, window_sec: float = 30) -> tuple[float, str]:
        """检测价格动量 (短期单边移动)"""
        if len(self.recent_trades) < 3:
            return 0.0, "neutral"

        now = self.recent_trades[-1][0]
        cutoff = now - window_sec
        recent = [(t[0], t[4]) for t in self.recent_trades if t[0] > cutoff]  # (ts, mid)

        if len(recent) < 2:
            return 0.0, "neutral"

        start_mid = recent[0][1]
        end_mid = recent[-1][1]
        move = end_mid - start_mid

        direction = "up" if move > 0 else "down" if move < 0 else "neutral"
        return abs(move), direction

    def get_trade_imbalance(self, window_sec: float = 60) -> float:
        """计算最近交易的买卖不平衡 (-1 到 1, 正为买方主导)"""
        if not self.recent_trades:
            return 0.0

        now = self.recent_trades[-1][0]
        cutoff = now - window_sec
        recent = [t for t in self.recent_trades if t[0] > cutoff]

        buy_vol = sum(t[2] for t in recent if t[3] == "BUY")
        sell_vol = sum(t[2] for t in recent if t[3] == "SELL")
        total = buy_vol + sell_vol

        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total

    def is_toxic_flow(self, side: str) -> tuple[bool, str]:
        """判断当前订单流是否对我方有毒"""
        vpin = self.get_vpin()
        momentum, direction = self.get_momentum()
        imbalance = self.get_trade_imbalance()

        # 对于BUY订单，卖方主导的toxic flow是危险的(价格可能下跌)
        # 对于SELL订单，买方主导的toxic flow是危险的(价格可能上涨)
        if side == "BUY":
            if vpin > 0.35 and imbalance < -0.3:
                return True, f"VPIN={vpin:.2f}, sell_dominant={imbalance:.2f}"
            if direction == "down" and momentum > 0.02:
                return True, f"momentum_down={momentum:.3f}"
        else:  # SELL
            if vpin > 0.35 and imbalance > 0.3:
                return True, f"VPIN={vpin:.2f}, buy_dominant={imbalance:.2f}"
            if direction == "up" and momentum > 0.02:
                return True, f"momentum_up={momentum:.3f}"

        return False, ""


@dataclass
class MarketTracker:
    """State for a single market being paper-traded."""
    condition_id: str
    token_id: str
    question: str
    daily_rate: float
    min_size: int
    max_spread: float
    # Virtual orders
    bid_order: VirtualOrder | None = None
    ask_order: VirtualOrder | None = None
    # Position tracking
    yes_position: float = 0.0
    yes_avg_cost: float = 0.0
    cash_delta: float = 0.0  # running cash change from fills
    # Stats
    total_bid_fills: int = 0
    total_ask_fills: int = 0
    total_trades_seen: int = 0
    last_trade_id: str = ""
    # Timing
    bid_cooldown_until: float = 0.0
    ask_cooldown_until: float = 0.0
    last_ob_mid: float = 0.0
    price_history: list[tuple[float, float]] = field(default_factory=list)  # (ts, mid)
    # === 新增: 高级追踪 ===
    informed_stats: InformedTraderStats = field(default_factory=InformedTraderStats)
    order_sequence: int = 0  # 订单序列号
    # 队列追踪: {price_level: cumulative_volume_filled}
    bid_level_fills: dict = field(default_factory=dict)
    ask_level_fills: dict = field(default_factory=dict)
    # 回测校准数据
    fills_by_time_in_queue: list = field(default_factory=list)  # (queue_time_sec, was_filled)
    toxic_fill_count: int = 0  # 被toxic flow吃的fill数量


# ── Database ──

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_orders (
            order_id TEXT PRIMARY KEY,
            condition_id TEXT,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            placed_at REAL,
            mid_at_place REAL,
            status TEXT,
            filled_at REAL,
            fill_price REAL,
            mid_at_fill REAL,
            mid_after_60s REAL,
            queue_depth REAL,
            -- 新增: 队列位置追踪
            queue_position REAL DEFAULT 0,
            queue_time_sec REAL DEFAULT 0,
            is_toxic INTEGER DEFAULT 0,
            vpin_at_fill REAL DEFAULT 0,
            imbalance_at_fill REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS paper_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            condition_id TEXT,
            mid REAL,
            spread_cents REAL,
            bid_depth_5 REAL,
            ask_depth_5 REAL,
            total_q REAL,
            n_trades_since_last INTEGER,
            -- 新增: 竞争和流动性指标
            vpin REAL DEFAULT 0,
            trade_imbalance REAL DEFAULT 0,
            q_competition REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS paper_trades_seen (
            trade_id TEXT PRIMARY KEY,
            condition_id TEXT,
            ts REAL,
            price REAL,
            size REAL,
            side TEXT,
            mid_at_trade REAL DEFAULT 0
        );
        -- 新增: 队列时间分布追踪
        CREATE TABLE IF NOT EXISTS queue_time_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            condition_id TEXT,
            queue_time_sec REAL,
            was_filled INTEGER,
            recorded_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_po_cid ON paper_orders(condition_id);
        CREATE INDEX IF NOT EXISTS idx_ps_cid ON paper_snapshots(condition_id, ts);
        CREATE INDEX IF NOT EXISTS idx_qts_cid ON queue_time_stats(condition_id);
    """)

    # 添加新列到现有表(如果不存在)
    try:
        conn.execute("ALTER TABLE paper_orders ADD COLUMN queue_position REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_orders ADD COLUMN queue_time_sec REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_orders ADD COLUMN is_toxic INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_orders ADD COLUMN vpin_at_fill REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_orders ADD COLUMN imbalance_at_fill REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_snapshots ADD COLUMN vpin REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_snapshots ADD COLUMN trade_imbalance REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_snapshots ADD COLUMN q_competition REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE paper_trades_seen ADD COLUMN mid_at_trade REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    return conn


# ── Core Paper Trading Logic ──

class PaperTrader:
    """Simulates order placement and tracks fills against real data."""

    def __init__(self, config: MantisConfig, client: PolymarketClient, conn: sqlite3.Connection):
        self.config = config
        self.client = client
        self.conn = conn
        self.trackers: dict[str, MarketTracker] = {}
        self._order_counter = 0
        self._pending_fill_checks: list[tuple[float, str, float]] = []
        # (check_at_ts, order_id, mid_at_fill)

    def add_market(self, market_info: dict):
        """Start tracking a market."""
        cid = market_info["condition_id"]
        self.trackers[cid] = MarketTracker(
            condition_id=cid,
            token_id=market_info["token_id"],
            question=market_info["question"],
            daily_rate=market_info["daily_rate"],
            min_size=market_info["min_size"],
            max_spread=market_info["max_spread"],
        )
        logger.info(f"Tracking: {market_info['question'][:50]}")

    def tick(self):
        """One cycle: fetch data, check fills, place/refresh virtual orders."""
        now = time.time()

        # Check deferred adverse selection measurements
        self._check_deferred_measurements(now)

        for cid, tracker in self.trackers.items():
            try:
                self._tick_market(tracker, now)
            except Exception as e:
                logger.debug(f"Tick error for {cid[:8]}: {e}")

    def _tick_market(self, tracker: MarketTracker, now: float):
        """Process one market."""
        # 1. Fetch orderbook
        ob = self.client.fetch_orderbook(tracker.token_id)
        if not ob.bids or not ob.asks:
            return

        mid = ob.midpoint
        tracker.last_ob_mid = mid
        tracker.price_history.append((now, mid))
        # Keep last 2 hours of price history
        cutoff = now - 7200
        tracker.price_history = [(t, p) for t, p in tracker.price_history if t > cutoff]

        # Record snapshot
        bid_depth_5 = sum(l.size for l in ob.bids[:5])
        ask_depth_5 = sum(l.size for l in ob.asks[:5])
        total_q = compute_total_q_min(ob, tracker.max_spread)

        # 2. Fetch recent trades via public Data API
        trades = self._fetch_public_trades(tracker.condition_id)
        new_trades = self._filter_new_trades(tracker, trades, mid)
        tracker.total_trades_seen += len(new_trades)

        # Check if any of our virtual orders would have been filled
        self._check_fills(tracker, new_trades, mid, now)

        # 获取informed trader统计
        vpin = tracker.informed_stats.get_vpin()
        imbalance = tracker.informed_stats.get_trade_imbalance()

        # Record snapshot with extended data
        self.conn.execute(
            "INSERT INTO paper_snapshots (ts, condition_id, mid, spread_cents, "
            "bid_depth_5, ask_depth_5, total_q, n_trades_since_last, "
            "vpin, trade_imbalance, q_competition) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, tracker.condition_id, mid, ob.spread_cents,
             bid_depth_5, ask_depth_5, total_q, len(new_trades),
             vpin, imbalance, total_q),
        )

        # 3. Place or refresh virtual orders
        self._manage_orders(tracker, ob, mid, now)

        self.conn.commit()

    def _fetch_public_trades(self, condition_id: str) -> list[dict]:
        """Fetch recent trades from the public Data API (no auth needed)."""
        import httpx
        try:
            resp = httpx.get(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": "50"},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json() or []
            # Normalize field names for consistency
            trades = []
            for t in raw:
                trades.append({
                    "id": f"{t.get('proxyWallet', '')}-{t.get('timestamp', '')}",
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "side": t.get("side", "").upper(),
                    "timestamp": t.get("timestamp", 0),
                })
            return trades
        except Exception as e:
            logger.debug(f"Failed to fetch trades: {e}")
            return []

    def _filter_new_trades(self, tracker: MarketTracker, trades: list[dict],
                           current_mid: float = 0) -> list[dict]:
        """Filter trades we haven't seen yet."""
        new = []
        seen_ids = set()
        # Get already-seen trade IDs from DB (batch check)
        if trades:
            trade_ids = [t.get("id", "") for t in trades]
            placeholders = ",".join("?" * len(trade_ids))
            rows = self.conn.execute(
                f"SELECT trade_id FROM paper_trades_seen WHERE trade_id IN ({placeholders})",
                trade_ids,
            ).fetchall()
            seen_ids = {r[0] for r in rows}

        for t in trades:
            tid = t.get("id", "")
            if not tid or tid in seen_ids:
                continue
            new.append(t)
            # Record this trade with mid price
            self.conn.execute(
                "INSERT OR IGNORE INTO paper_trades_seen "
                "(trade_id, condition_id, ts, price, size, side, mid_at_trade) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (tid, tracker.condition_id, time.time(),
                 float(t.get("price", 0)), float(t.get("size", 0)),
                 t.get("side", ""), current_mid),
            )
        return new

    def _check_fills(self, tracker: MarketTracker, new_trades: list[dict],
                     current_mid: float, now: float):
        """Check if new trades would have filled our virtual orders.

        改进的Fill逻辑 (Time Priority + Pro-rata混合模型):
        1. Time Priority: 追踪我们在队列中的位置，前面的订单需要先被fill
        2. Pro-rata: 当同一tick有多笔trade时，按比例分配
        3. Toxic Flow Detection: 检测是否被informed trader吃单

        Fill logic: A trade crosses our virtual order if:
        - For our BUY order: a SELL trade at price <= our bid price
        - For our ASK order: a BUY trade at price >= our ask price
        """
        import random

        for trade in new_trades:
            t_price = float(trade.get("price", 0))
            t_size = float(trade.get("size", 0))
            t_side = trade.get("side", "").upper()
            t_ts = trade.get("timestamp", now)

            # 更新informed trader统计
            tracker.informed_stats.add_trade(t_ts, t_price, t_size, t_side, current_mid)

            # Check bid fill: someone sold at or below our bid
            if tracker.bid_order and tracker.bid_order.status == "live":
                order = tracker.bid_order
                if t_side == "SELL" and t_price <= order.price:
                    # 更新该价位的累计成交量
                    price_key = round(order.price, 4)
                    prev_fill_vol = tracker.bid_level_fills.get(price_key, 0)
                    tracker.bid_level_fills[price_key] = prev_fill_vol + t_size

                    # Time Priority 模型: 我们需要等前面的队列被消耗
                    # queue_position 是我们前面的量，需要先被fill
                    volume_filled_since_join = tracker.bid_level_fills[price_key] - order.volume_filled_at_level

                    if volume_filled_since_join >= order.queue_position:
                        # 前面的队列已经被消耗，现在检查我们是否被fill
                        remaining_volume = volume_filled_since_join - order.queue_position
                        fill_prob = self._compute_fill_probability(
                            order, remaining_volume, t_size, tracker, "BUY"
                        )
                        if random.random() < fill_prob:
                            # 检测是否是toxic fill
                            is_toxic, toxic_reason = tracker.informed_stats.is_toxic_flow("BUY")
                            if is_toxic:
                                tracker.toxic_fill_count += 1
                                logger.debug(f"Toxic fill detected: {toxic_reason}")

                            queue_time = now - order.placed_at
                            tracker.fills_by_time_in_queue.append((queue_time, True))
                            self._record_fill(tracker, order, t_price, current_mid, now,
                                              is_toxic=is_toxic, toxic_reason=toxic_reason)
                    else:
                        # 记录队列推进但未fill
                        order.volume_filled_at_level = tracker.bid_level_fills[price_key]

            # Check ask fill: someone bought at or above our ask
            if tracker.ask_order and tracker.ask_order.status == "live":
                order = tracker.ask_order
                if t_side == "BUY" and t_price >= order.price:
                    price_key = round(order.price, 4)
                    prev_fill_vol = tracker.ask_level_fills.get(price_key, 0)
                    tracker.ask_level_fills[price_key] = prev_fill_vol + t_size

                    volume_filled_since_join = tracker.ask_level_fills[price_key] - order.volume_filled_at_level

                    if volume_filled_since_join >= order.queue_position:
                        remaining_volume = volume_filled_since_join - order.queue_position
                        fill_prob = self._compute_fill_probability(
                            order, remaining_volume, t_size, tracker, "SELL"
                        )
                        if random.random() < fill_prob:
                            is_toxic, toxic_reason = tracker.informed_stats.is_toxic_flow("SELL")
                            if is_toxic:
                                tracker.toxic_fill_count += 1

                            queue_time = now - order.placed_at
                            tracker.fills_by_time_in_queue.append((queue_time, True))
                            self._record_fill(tracker, order, t_price, current_mid, now,
                                              is_toxic=is_toxic, toxic_reason=toxic_reason)
                    else:
                        order.volume_filled_at_level = tracker.ask_level_fills[price_key]

    def _compute_fill_probability(self, order: VirtualOrder, remaining_volume: float,
                                   trade_size: float, tracker: MarketTracker,
                                   side: str) -> float:
        """计算fill概率 (混合Time Priority + Pro-rata).

        Polymarket使用Pro-rata模型，但在实际中:
        1. 较早的订单有轻微优势
        2. 较大的taker订单可能partial fill我们
        3. 如果taker订单远大于队列，fill概率接近1
        """
        our_size = order.size
        queue_remaining = max(0, order.queue_position - remaining_volume + our_size)

        if queue_remaining <= 0:
            return 1.0

        # 基础 Pro-rata 概率
        base_prob = our_size / queue_remaining if queue_remaining > 0 else 1.0

        # 调整因子1: taker订单大小
        # 如果taker订单很大，我们更可能被fill
        size_multiplier = min(2.0, trade_size / (our_size + 10))

        # 调整因子2: 时间在队列中
        # 在Polymarket没有时间优先，但市场maker可能有re-quote行为
        # 导致长时间在队列中的订单更可能被fill
        time_in_queue = time.time() - order.placed_at
        time_bonus = min(0.2, time_in_queue / 600)  # 最多+20%，10分钟达到

        # 最终概率
        prob = min(1.0, base_prob * (1 + size_multiplier * 0.3) + time_bonus)

        return prob

    def _record_fill(self, tracker: MarketTracker, order: VirtualOrder,
                     fill_price: float, current_mid: float, now: float,
                     is_toxic: bool = False, toxic_reason: str = ""):
        """Record a virtual fill with toxic flow detection."""
        order.status = "filled"
        order.filled_at = now
        order.fill_price = fill_price
        order.mid_at_fill = current_mid

        # Update position
        if order.side == "BUY":
            cost = order.size * fill_price
            tracker.cash_delta -= cost
            total_value = tracker.yes_avg_cost * tracker.yes_position + fill_price * order.size
            tracker.yes_position += order.size
            tracker.yes_avg_cost = total_value / tracker.yes_position if tracker.yes_position > 0 else 0
            tracker.total_bid_fills += 1
            tracker.bid_order = None
            tracker.bid_cooldown_until = now + self.config.engine.post_fill_cooldown_sec
        else:
            revenue = order.size * fill_price
            tracker.cash_delta += revenue
            tracker.yes_position = max(0, tracker.yes_position - order.size)
            tracker.total_ask_fills += 1
            tracker.ask_order = None
            tracker.ask_cooldown_until = now + self.config.engine.post_fill_cooldown_sec

        # Schedule adverse selection check (60s later)
        self._pending_fill_checks.append((now + 60, order.order_id, current_mid))

        # 计算额外的统计信息
        queue_time = now - order.placed_at
        vpin = tracker.informed_stats.get_vpin()
        imbalance = tracker.informed_stats.get_trade_imbalance()

        # Save to DB (extended schema)
        self.conn.execute(
            "INSERT OR REPLACE INTO paper_orders "
            "(order_id, condition_id, token_id, side, price, size, placed_at, "
            "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, queue_depth, "
            "queue_position, queue_time_sec, is_toxic, vpin_at_fill, imbalance_at_fill) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (order.order_id, tracker.condition_id, order.token_id,
             order.side, order.price, order.size, order.placed_at,
             order.mid_at_place, "filled", order.filled_at, fill_price,
             current_mid, 0.0, order.queue_depth_at_place,
             order.queue_position, queue_time, int(is_toxic), vpin, imbalance),
        )

        toxic_flag = " [TOXIC]" if is_toxic else ""
        logger.info(
            f"FILL: {order.side} {order.size}@{fill_price:.4f} "
            f"mid={current_mid:.4f} queue={queue_time:.0f}s{toxic_flag} | {tracker.question[:30]}"
        )

    def _check_deferred_measurements(self, now: float):
        """Check 60s-after-fill price for adverse selection measurement."""
        remaining = []
        for check_ts, order_id, mid_at_fill in self._pending_fill_checks:
            if now >= check_ts:
                # Find the market this order belongs to
                row = self.conn.execute(
                    "SELECT condition_id FROM paper_orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
                if row:
                    cid = row[0]
                    tracker = self.trackers.get(cid)
                    if tracker:
                        current_mid = tracker.last_ob_mid
                        self.conn.execute(
                            "UPDATE paper_orders SET mid_after_60s = ? WHERE order_id = ?",
                            (current_mid, order_id),
                        )
                        move = current_mid - mid_at_fill
                        logger.info(
                            f"60s check: order {order_id[:8]} mid moved "
                            f"{move:+.4f} ({move/mid_at_fill*100:+.2f}%)"
                        )
            else:
                remaining.append((check_ts, order_id, mid_at_fill))
        self._pending_fill_checks = remaining

    def _manage_orders(self, tracker: MarketTracker, ob, mid: float, now: float):
        """Place or refresh virtual orders using defiance_cr strategy."""
        max_spread = tracker.max_spread
        dist_c = max_spread * self.config.engine.reward_spread_pct
        half = dist_c / 100  # cents to price units
        half = max(half, 0.001)
        if half * 100 >= max_spread:
            half = (max_spread - 0.1) / 100

        bid_price = round(mid - half, 4)
        ask_price = round(mid + half, 4)
        bid_price = max(0.001, bid_price)
        ask_price = min(0.999, ask_price)

        min_size = tracker.min_size

        # Estimate queue depth at our price level (真正的队列位置)
        bid_queue = self._estimate_queue_depth(ob.bids, bid_price, "bid")
        ask_queue = self._estimate_queue_depth(ob.asks, ask_price, "ask")

        # 计算Q竞争 (同一价位的Q score竞争)
        q_competition = self._estimate_q_competition(ob, bid_price, ask_price, tracker.max_spread)

        # Refresh bid
        if tracker.bid_order and tracker.bid_order.status == "live":
            # Reprice if mid moved significantly
            if abs(bid_price - tracker.bid_order.price) > self.config.engine.reprice_threshold:
                # 记录cancelled order的队列时间
                queue_time = now - tracker.bid_order.placed_at
                tracker.fills_by_time_in_queue.append((queue_time, False))
                tracker.bid_order.status = "cancelled"
                tracker.bid_order = None
            # Expire orders older than 5 minutes
            elif now - tracker.bid_order.placed_at > 300:
                queue_time = now - tracker.bid_order.placed_at
                tracker.fills_by_time_in_queue.append((queue_time, False))
                tracker.bid_order.status = "cancelled"
                tracker.bid_order = None

        if tracker.ask_order and tracker.ask_order.status == "live":
            if abs(ask_price - tracker.ask_order.price) > self.config.engine.reprice_threshold:
                queue_time = now - tracker.ask_order.placed_at
                tracker.fills_by_time_in_queue.append((queue_time, False))
                tracker.ask_order.status = "cancelled"
                tracker.ask_order = None
            elif now - tracker.ask_order.placed_at > 300:
                queue_time = now - tracker.ask_order.placed_at
                tracker.fills_by_time_in_queue.append((queue_time, False))
                tracker.ask_order.status = "cancelled"
                tracker.ask_order = None

        # Place new bid if needed
        if tracker.bid_order is None and now >= tracker.bid_cooldown_until:
            self._order_counter += 1
            tracker.order_sequence += 1
            oid = f"paper-bid-{self._order_counter}"

            # 获取当前价位的累计成交量(用于追踪队列推进)
            price_key = round(bid_price, 4)
            current_level_fills = tracker.bid_level_fills.get(price_key, 0)

            tracker.bid_order = VirtualOrder(
                order_id=oid,
                condition_id=tracker.condition_id,
                token_id=tracker.token_id,
                side="BUY",
                price=bid_price,
                size=min_size,
                placed_at=now,
                mid_at_place=mid,
                queue_depth_at_place=bid_queue,
                queue_position=bid_queue,  # 我们排在当前队列后面
                volume_filled_at_level=current_level_fills,
                join_sequence=tracker.order_sequence,
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO paper_orders "
                "(order_id, condition_id, token_id, side, price, size, placed_at, "
                "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, "
                "queue_depth, queue_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, tracker.condition_id, tracker.token_id,
                 "BUY", bid_price, min_size, now, mid, "live", 0, 0, 0, 0, bid_queue, bid_queue),
            )

        # Place new ask if needed
        if tracker.ask_order is None and now >= tracker.ask_cooldown_until:
            self._order_counter += 1
            tracker.order_sequence += 1
            oid = f"paper-ask-{self._order_counter}"

            price_key = round(ask_price, 4)
            current_level_fills = tracker.ask_level_fills.get(price_key, 0)

            tracker.ask_order = VirtualOrder(
                order_id=oid,
                condition_id=tracker.condition_id,
                token_id=tracker.token_id,
                side="SELL",
                price=ask_price,
                size=min_size,
                placed_at=now,
                mid_at_place=mid,
                queue_depth_at_place=ask_queue,
                queue_position=ask_queue,
                volume_filled_at_level=current_level_fills,
                join_sequence=tracker.order_sequence,
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO paper_orders "
                "(order_id, condition_id, token_id, side, price, size, placed_at, "
                "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, "
                "queue_depth, queue_position) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, tracker.condition_id, tracker.token_id,
                 "SELL", ask_price, min_size, now, mid, "live", 0, 0, 0, 0, ask_queue, ask_queue),
            )

    def _estimate_q_competition(self, ob, bid_price: float, ask_price: float,
                                 max_spread: float) -> float:
        """估算Q score竞争程度.

        计算在我们报价价位附近的其他maker的Q贡献。
        Q竞争越高，我们获得的奖励占比越低。
        """
        from .scanner import compute_total_q_min
        total_q = compute_total_q_min(ob, max_spread)
        return total_q

    def _estimate_queue_depth(self, levels: list, our_price: float, side: str) -> float:
        """Estimate how much size is queued ahead of us at our price level.

        For bids: sum of all bid sizes at prices >= our price (they're ahead in queue)
        For asks: sum of all ask sizes at prices <= our price
        """
        depth = 0.0
        for level in levels:
            if side == "bid" and level.price >= our_price:
                depth += level.size
            elif side == "ask" and level.price <= our_price:
                depth += level.size
        return depth


# ── Statistics & Calibration ──

def get_stats(db_path: Path = DB_PATH) -> str:
    """Generate statistics report from paper trading data."""
    if not db_path.exists():
        return "No paper trading data yet. Run the paper trader first."

    conn = sqlite3.connect(str(db_path))
    lines = []

    # Overall stats
    total_orders = conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0]
    filled_orders = conn.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE status='filled'"
    ).fetchone()[0]
    live_orders = conn.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE status='live'"
    ).fetchone()[0]

    lines.append("=" * 60)
    lines.append("PAPER TRADING STATISTICS (Enhanced)")
    lines.append("=" * 60)
    lines.append(f"Total orders placed:  {total_orders}")
    lines.append(f"Filled:               {filled_orders}")
    lines.append(f"Still live:           {live_orders}")
    fill_rate = filled_orders / total_orders * 100 if total_orders > 0 else 0
    lines.append(f"Fill rate:            {fill_rate:.1f}%")
    lines.append("")

    # Fill rate by side
    for side in ["BUY", "SELL"]:
        total = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE side=?", (side,)
        ).fetchone()[0]
        filled = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE side=? AND status='filled'", (side,)
        ).fetchone()[0]
        rate = filled / total * 100 if total > 0 else 0
        lines.append(f"  {side}: {filled}/{total} = {rate:.1f}%")

    lines.append("")

    # === 新增: Queue Time Analysis ===
    lines.append("--- Queue Time Analysis ---")
    rows = conn.execute(
        "SELECT queue_time_sec FROM paper_orders WHERE status='filled' AND queue_time_sec > 0"
    ).fetchall()
    if rows:
        queue_times = [r[0] for r in rows]
        avg_qt = sum(queue_times) / len(queue_times)
        min_qt = min(queue_times)
        max_qt = max(queue_times)
        median_qt = sorted(queue_times)[len(queue_times) // 2]
        lines.append(f"  Avg queue time:    {avg_qt:.1f}s")
        lines.append(f"  Median:            {median_qt:.1f}s")
        lines.append(f"  Min/Max:           {min_qt:.1f}s / {max_qt:.1f}s")

        # Queue time buckets
        buckets = [0, 30, 60, 120, 300]
        for i, b in enumerate(buckets[:-1]):
            count = sum(1 for qt in queue_times if b <= qt < buckets[i+1])
            pct = count / len(queue_times) * 100
            lines.append(f"    {b}-{buckets[i+1]}s: {count} ({pct:.0f}%)")
        count_long = sum(1 for qt in queue_times if qt >= 300)
        pct_long = count_long / len(queue_times) * 100
        lines.append(f"    300s+: {count_long} ({pct_long:.0f}%)")
    else:
        lines.append("  No queue time data yet")

    lines.append("")

    # === 新增: Toxic Flow Analysis ===
    lines.append("--- Toxic Flow Detection ---")
    try:
        toxic_fills = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE status='filled' AND is_toxic=1"
        ).fetchone()[0]
        total_fills = filled_orders
        toxic_pct = toxic_fills / total_fills * 100 if total_fills > 0 else 0
        lines.append(f"  Toxic fills:       {toxic_fills}/{total_fills} ({toxic_pct:.1f}%)")

        # VPIN correlation with adverse selection
        rows = conn.execute(
            "SELECT vpin_at_fill, mid_at_fill, mid_after_60s, side "
            "FROM paper_orders WHERE status='filled' AND mid_after_60s > 0 AND vpin_at_fill > 0"
        ).fetchall()
        if rows:
            high_vpin_adverse = 0
            low_vpin_adverse = 0
            high_vpin_count = 0
            low_vpin_count = 0
            for vpin, mid_fill, mid_60s, side in rows:
                move = mid_60s - mid_fill
                is_adverse = (side == "BUY" and move < 0) or (side == "SELL" and move > 0)
                if vpin > 0.3:
                    high_vpin_count += 1
                    if is_adverse:
                        high_vpin_adverse += 1
                else:
                    low_vpin_count += 1
                    if is_adverse:
                        low_vpin_adverse += 1

            high_rate = high_vpin_adverse / high_vpin_count * 100 if high_vpin_count > 0 else 0
            low_rate = low_vpin_adverse / low_vpin_count * 100 if low_vpin_count > 0 else 0
            lines.append(f"  High VPIN (>0.3) adverse rate: {high_rate:.1f}% ({high_vpin_count} fills)")
            lines.append(f"  Low VPIN (<=0.3) adverse rate: {low_rate:.1f}% ({low_vpin_count} fills)")
    except sqlite3.OperationalError:
        lines.append("  (toxic tracking columns not available)")

    lines.append("")

    # Adverse selection analysis
    lines.append("--- Adverse Selection (60s post-fill) ---")
    rows = conn.execute(
        "SELECT side, price, mid_at_fill, mid_after_60s "
        "FROM paper_orders WHERE status='filled' AND mid_after_60s > 0"
    ).fetchall()

    if rows:
        adverse_count = 0
        total_adverse_move = 0.0
        adverse_moves = []
        for side, price, mid_fill, mid_60s in rows:
            move = mid_60s - mid_fill
            if side == "BUY" and move < 0:
                adverse_count += 1
                adverse_moves.append(abs(move))
                total_adverse_move += abs(move)
            elif side == "SELL" and move > 0:
                adverse_count += 1
                adverse_moves.append(abs(move))
                total_adverse_move += abs(move)

        adverse_pct = adverse_count / len(rows) * 100
        avg_adverse = total_adverse_move / adverse_count if adverse_count > 0 else 0
        lines.append(f"  Fills with 60s data:  {len(rows)}")
        lines.append(f"  Adverse moves:        {adverse_count} ({adverse_pct:.1f}%)")
        lines.append(f"  Avg adverse move:     {avg_adverse:.4f}")

        # Adverse move distribution
        if adverse_moves:
            percentiles = [50, 75, 90, 95]
            sorted_moves = sorted(adverse_moves)
            for p in percentiles:
                idx = int(len(sorted_moves) * p / 100)
                val = sorted_moves[min(idx, len(sorted_moves)-1)]
                lines.append(f"    P{p} adverse move: {val:.4f}")
    else:
        lines.append("  No fill data with 60s measurements yet")

    lines.append("")

    # Queue depth stats
    lines.append("--- Queue Depth at Fill ---")
    rows = conn.execute(
        "SELECT AVG(queue_depth), MIN(queue_depth), MAX(queue_depth), "
        "AVG(size), AVG(queue_position) FROM paper_orders WHERE status='filled'"
    ).fetchone()
    if rows and rows[0] is not None:
        avg_q, min_q, max_q, avg_size, avg_pos = rows
        lines.append(f"  Avg queue depth:    {avg_q:.0f}")
        lines.append(f"  Avg queue position: {avg_pos or 0:.0f}")
        lines.append(f"  Min/Max depth:      {min_q:.0f} / {max_q:.0f}")
        lines.append(f"  Avg order size:     {avg_size:.0f}")
        if avg_q > 0:
            implied_fill_prob = avg_size / (avg_size + avg_q)
            lines.append(f"  Implied fill prob:  {implied_fill_prob:.1%}")

    lines.append("")

    # Per-market breakdown
    lines.append("--- Per Market ---")
    markets = conn.execute(
        "SELECT condition_id, COUNT(*) as total, "
        "SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as fills "
        "FROM paper_orders GROUP BY condition_id"
    ).fetchall()

    for cid, total, fills in markets:
        rate = fills / total * 100 if total > 0 else 0
        lines.append(f"  {cid[:12]}... orders={total} fills={fills} rate={rate:.1f}%")

    lines.append("")

    # Trade frequency
    lines.append("--- Market Trade Frequency ---")
    snap_rows = conn.execute(
        "SELECT condition_id, COUNT(*), SUM(n_trades_since_last), "
        "MIN(ts), MAX(ts) FROM paper_snapshots GROUP BY condition_id"
    ).fetchall()
    for cid, n_snaps, total_trades, t0, t1 in snap_rows:
        hours = (t1 - t0) / 3600 if t1 > t0 else 1
        trades_per_hour = (total_trades or 0) / hours
        lines.append(f"  {cid[:12]}... {trades_per_hour:.1f} trades/hr ({n_snaps} snapshots)")

    lines.append("")

    # === 新增: VPIN Statistics ===
    lines.append("--- VPIN Statistics ---")
    try:
        snap_rows = conn.execute(
            "SELECT AVG(vpin), MIN(vpin), MAX(vpin) FROM paper_snapshots WHERE vpin > 0"
        ).fetchone()
        if snap_rows and snap_rows[0]:
            avg_vpin, min_vpin, max_vpin = snap_rows
            lines.append(f"  Avg VPIN:     {avg_vpin:.3f}")
            lines.append(f"  Min/Max:      {min_vpin:.3f} / {max_vpin:.3f}")

            # VPIN distribution
            high_vpin = conn.execute(
                "SELECT COUNT(*) FROM paper_snapshots WHERE vpin > 0.35"
            ).fetchone()[0]
            total_snaps = conn.execute(
                "SELECT COUNT(*) FROM paper_snapshots WHERE vpin > 0"
            ).fetchone()[0]
            high_pct = high_vpin / total_snaps * 100 if total_snaps > 0 else 0
            lines.append(f"  High VPIN (>0.35): {high_pct:.1f}% of snapshots")
    except sqlite3.OperationalError:
        lines.append("  (VPIN columns not available)")

    conn.close()
    return "\n".join(lines)


def get_calibration(db_path: Path = DB_PATH) -> dict:
    """Extract calibrated parameters for realistic backtesting.

    Returns a dict that can be plugged directly into the backtest model.
    Enhanced with queue time, toxic flow, and VPIN metrics.
    """
    if not db_path.exists():
        return {"error": "No data. Run paper trader first."}

    conn = sqlite3.connect(str(db_path))
    result = {}

    # 1. Fill probability
    total = conn.execute("SELECT COUNT(*) FROM paper_orders WHERE status != 'live'").fetchone()[0]
    filled = conn.execute("SELECT COUNT(*) FROM paper_orders WHERE status='filled'").fetchone()[0]
    result["raw_fill_rate"] = filled / total if total > 0 else 0

    # Pro-rata implied
    row = conn.execute(
        "SELECT AVG(size), AVG(queue_depth) FROM paper_orders WHERE status='filled'"
    ).fetchone()
    if row and row[0]:
        avg_size, avg_queue = row
        result["avg_order_size"] = avg_size
        result["avg_queue_depth"] = avg_queue
        result["pro_rata_fill_prob"] = avg_size / (avg_size + avg_queue) if avg_queue > 0 else 1.0
    else:
        result["pro_rata_fill_prob"] = 0.15  # fallback

    # 2. Adverse selection rate
    rows = conn.execute(
        "SELECT side, mid_at_fill, mid_after_60s "
        "FROM paper_orders WHERE status='filled' AND mid_after_60s > 0"
    ).fetchall()
    if rows:
        adverse = 0
        moves = []
        for side, mid_fill, mid_60s in rows:
            move = mid_60s - mid_fill
            if (side == "BUY" and move < 0) or (side == "SELL" and move > 0):
                adverse += 1
                moves.append(abs(move))
        result["adverse_selection_rate"] = adverse / len(rows)
        result["avg_adverse_move"] = sum(moves) / len(moves) if moves else 0
        # 添加adverse move分布
        if moves:
            sorted_moves = sorted(moves)
            result["adverse_move_p50"] = sorted_moves[len(sorted_moves) // 2]
            result["adverse_move_p90"] = sorted_moves[int(len(sorted_moves) * 0.9)]
            result["adverse_move_p95"] = sorted_moves[int(len(sorted_moves) * 0.95)]
    else:
        result["adverse_selection_rate"] = 0.30  # fallback
        result["avg_adverse_move"] = 0.005

    # 3. Trade frequency per market
    snap_rows = conn.execute(
        "SELECT condition_id, SUM(n_trades_since_last), MIN(ts), MAX(ts) "
        "FROM paper_snapshots GROUP BY condition_id"
    ).fetchall()
    freqs = []
    for cid, total_trades, t0, t1 in snap_rows:
        hours = (t1 - t0) / 3600 if t1 > t0 else 1
        freqs.append((total_trades or 0) / hours)
    result["avg_trades_per_hour"] = sum(freqs) / len(freqs) if freqs else 5.0

    # 4. Spread stats
    row = conn.execute(
        "SELECT AVG(spread_cents), AVG(total_q) FROM paper_snapshots"
    ).fetchone()
    if row:
        result["avg_spread_cents"] = row[0] or 3.0
        result["avg_total_q"] = row[1] or 100.0

    # === 新增: Queue Time 校准 ===
    try:
        rows = conn.execute(
            "SELECT queue_time_sec FROM paper_orders WHERE status='filled' AND queue_time_sec > 0"
        ).fetchall()
        if rows:
            queue_times = [r[0] for r in rows]
            result["avg_queue_time_sec"] = sum(queue_times) / len(queue_times)
            sorted_qt = sorted(queue_times)
            result["queue_time_p50"] = sorted_qt[len(sorted_qt) // 2]
            result["queue_time_p90"] = sorted_qt[int(len(sorted_qt) * 0.9)]

            # Fill rate by queue time bucket
            short_fills = sum(1 for qt in queue_times if qt < 60)
            medium_fills = sum(1 for qt in queue_times if 60 <= qt < 180)
            long_fills = sum(1 for qt in queue_times if qt >= 180)
            result["fill_rate_by_queue_time"] = {
                "under_60s": short_fills / len(queue_times) if queue_times else 0,
                "60_180s": medium_fills / len(queue_times) if queue_times else 0,
                "over_180s": long_fills / len(queue_times) if queue_times else 0,
            }
    except sqlite3.OperationalError:
        pass

    # === 新增: Toxic Flow 校准 ===
    try:
        toxic_fills = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE status='filled' AND is_toxic=1"
        ).fetchone()[0]
        result["toxic_fill_rate"] = toxic_fills / filled if filled > 0 else 0

        # VPIN和adverse selection的关联
        rows = conn.execute(
            "SELECT vpin_at_fill, mid_at_fill, mid_after_60s, side "
            "FROM paper_orders WHERE status='filled' AND mid_after_60s > 0 AND vpin_at_fill > 0"
        ).fetchall()
        if rows:
            high_vpin_adverse = 0
            high_vpin_count = 0
            low_vpin_adverse = 0
            low_vpin_count = 0
            for vpin, mid_fill, mid_60s, side in rows:
                move = mid_60s - mid_fill
                is_adverse = (side == "BUY" and move < 0) or (side == "SELL" and move > 0)
                if vpin > 0.3:
                    high_vpin_count += 1
                    if is_adverse:
                        high_vpin_adverse += 1
                else:
                    low_vpin_count += 1
                    if is_adverse:
                        low_vpin_adverse += 1

            result["adverse_rate_high_vpin"] = high_vpin_adverse / high_vpin_count if high_vpin_count > 0 else 0
            result["adverse_rate_low_vpin"] = low_vpin_adverse / low_vpin_count if low_vpin_count > 0 else 0
    except sqlite3.OperationalError:
        pass

    # === 新增: VPIN 统计 ===
    try:
        row = conn.execute(
            "SELECT AVG(vpin), AVG(trade_imbalance) FROM paper_snapshots WHERE vpin > 0"
        ).fetchone()
        if row and row[0]:
            result["avg_vpin"] = row[0]
            result["avg_trade_imbalance"] = row[1]
    except sqlite3.OperationalError:
        pass

    conn.close()

    # Enhanced Summary
    toxic_rate = result.get('toxic_fill_rate', 0)
    queue_time = result.get('avg_queue_time_sec', 0)
    result["_summary"] = (
        f"Fill prob: {result.get('pro_rata_fill_prob', 0):.1%} | "
        f"Adverse: {result.get('adverse_selection_rate', 0):.0%} | "
        f"Toxic: {toxic_rate:.0%} | "
        f"Queue: {queue_time:.0f}s | "
        f"Trades/hr: {result.get('avg_trades_per_hour', 0):.1f}"
    )

    return result


# ── Historical Replay Mode ──

def run_historical_replay(top_n: int = 5):
    """Download all available history and replay paper trading against it.

    Uses:
    - data-api.polymarket.com/trades for trade history (up to 500 per market)
    - clob.polymarket.com/prices-history for price series
    - clob.polymarket.com/book for current orderbook (queue depth snapshot)

    This gives immediate calibration data without waiting days.
    """
    import httpx
    import os
    import random

    os.chdir(str(_MODULE_DIR))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    config = load_config("config.yaml")
    client = PolymarketClient()

    from .collector import select_markets
    logger.info(f"Selecting top {top_n} markets for replay...")
    markets = select_markets(client, config, top_n)
    logger.info(f"Found {len(markets)} markets")

    # Use a separate DB for historical data
    hist_db_path = _MODULE_DIR / "data" / "paper_trades_hist.db"
    conn = init_db(hist_db_path)
    # Clear old data
    conn.executescript("DELETE FROM paper_orders; DELETE FROM paper_snapshots; DELETE FROM paper_trades_seen;")
    conn.commit()

    http = httpx.Client(timeout=15)
    all_results = []

    for idx, m in enumerate(markets):
        cid = m["condition_id"]
        token_id = m["token_id"]
        question = m["question"][:50]
        min_size = m["min_size"]
        max_spread = m["max_spread"]

        logger.info(f"\n--- [{idx+1}/{len(markets)}] {question} ---")

        # 1. Fetch current orderbook for queue depth estimation
        ob = client.fetch_orderbook(token_id)
        if not ob.bids or not ob.asks:
            logger.info("  Empty orderbook, skip")
            continue

        bid_depth_5 = sum(l.size for l in ob.bids[:5])
        ask_depth_5 = sum(l.size for l in ob.asks[:5])

        # 2. Fetch trade history
        trades = []
        r = http.get(f"{DATA_API_BASE}/trades",
                     params={"market": cid, "limit": "500"})
        if r.status_code == 200:
            raw = r.json() or []
            for t in raw:
                trades.append({
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "side": t.get("side", "").upper(),
                    "timestamp": int(t.get("timestamp", 0)),
                })
            trades.sort(key=lambda t: t["timestamp"])

        # 3. Fetch price history for mid prices
        r = http.get(f"https://clob.polymarket.com/prices-history",
                     params={"market": token_id, "interval": "max", "fidelity": "1"})
        price_history = []
        if r.status_code == 200:
            for pt_item in r.json().get("history", []):
                price_history.append((pt_item["t"], pt_item["p"]))

        logger.info(f"  {len(trades)} trades, {len(price_history)} price points")
        if not trades or not price_history:
            continue

        # 4. Build a time-indexed mid lookup
        def get_mid_at(ts):
            """Get interpolated mid price at timestamp."""
            if not price_history:
                return 0.5
            # Find nearest price point
            best = price_history[0][1]
            for pt_ts, pt_p in price_history:
                if pt_ts <= ts:
                    best = pt_p
                else:
                    break
            return best

        # 5. Simulate paper trading through history
        dist_c = max_spread * config.engine.reward_spread_pct
        half = max(dist_c / 100, 0.001)
        if half * 100 >= max_spread:
            half = (max_spread - 0.1) / 100

        # Estimate queue depth from current orderbook
        avg_queue = (bid_depth_5 + ask_depth_5) / 2

        bid_order_price = None
        ask_order_price = None
        bid_placed_mid = None
        ask_placed_mid = None
        cooldown_bid_until = 0
        cooldown_ask_until = 0

        fills = []
        total_orders = 0

        for trade in trades:
            ts = trade["timestamp"]
            t_price = trade["price"]
            t_size = trade["size"]
            t_side = trade["side"]
            mid = get_mid_at(ts)

            if mid <= 0 or mid >= 1:
                continue

            # Place/refresh virtual orders at current mid
            new_bid = round(mid - half, 4)
            new_ask = round(mid + half, 4)

            # Refresh orders every time mid changes significantly
            if bid_order_price is None or abs(new_bid - bid_order_price) > config.engine.reprice_threshold:
                if ts >= cooldown_bid_until:
                    bid_order_price = new_bid
                    bid_placed_mid = mid
                    total_orders += 1

            if ask_order_price is None or abs(new_ask - ask_order_price) > config.engine.reprice_threshold:
                if ts >= cooldown_ask_until:
                    ask_order_price = new_ask
                    ask_placed_mid = mid
                    total_orders += 1

            # Check fills
            if bid_order_price and t_side == "SELL" and t_price <= bid_order_price:
                fill_prob = min_size / (min_size + avg_queue) if avg_queue > 0 else 1.0
                if random.random() < fill_prob:
                    # Get mid 60s later
                    mid_60s = get_mid_at(ts + 60)
                    fills.append({
                        "side": "BUY", "price": bid_order_price,
                        "mid_at_fill": mid, "mid_after_60s": mid_60s,
                        "queue_depth": avg_queue, "size": min_size, "ts": ts,
                    })
                    # Save to DB
                    oid = f"hist-bid-{len(fills)}-{cid[:8]}"
                    conn.execute(
                        "INSERT OR REPLACE INTO paper_orders "
                        "(order_id, condition_id, token_id, side, price, size, placed_at, "
                        "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, queue_depth) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (oid, cid, token_id, "BUY", bid_order_price, min_size,
                         ts, bid_placed_mid or mid, "filled", ts, t_price, mid, mid_60s, avg_queue),
                    )
                    bid_order_price = None
                    cooldown_bid_until = ts + config.engine.post_fill_cooldown_sec

            if ask_order_price and t_side == "BUY" and t_price >= ask_order_price:
                fill_prob = min_size / (min_size + avg_queue) if avg_queue > 0 else 1.0
                if random.random() < fill_prob:
                    mid_60s = get_mid_at(ts + 60)
                    fills.append({
                        "side": "SELL", "price": ask_order_price,
                        "mid_at_fill": mid, "mid_after_60s": mid_60s,
                        "queue_depth": avg_queue, "size": min_size, "ts": ts,
                    })
                    oid = f"hist-ask-{len(fills)}-{cid[:8]}"
                    conn.execute(
                        "INSERT OR REPLACE INTO paper_orders "
                        "(order_id, condition_id, token_id, side, price, size, placed_at, "
                        "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, queue_depth) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (oid, cid, token_id, "SELL", ask_order_price, min_size,
                         ts, ask_placed_mid or mid, "filled", ts, t_price, mid, mid_60s, avg_queue),
                    )
                    ask_order_price = None
                    cooldown_ask_until = ts + config.engine.post_fill_cooldown_sec

        conn.commit()

        # Also record cancelled (non-filled) orders as estimate
        cancelled_orders = total_orders - len(fills)
        for i in range(min(cancelled_orders, 200)):  # cap DB size
            oid = f"hist-cancelled-{i}-{cid[:8]}"
            conn.execute(
                "INSERT OR IGNORE INTO paper_orders "
                "(order_id, condition_id, token_id, side, price, size, placed_at, "
                "mid_at_place, status, filled_at, fill_price, mid_at_fill, mid_after_60s, queue_depth) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (oid, cid, token_id, "BUY", 0, min_size, 0, 0, "cancelled", 0, 0, 0, 0, avg_queue),
            )
        conn.commit()

        # Per-market summary
        if trades:
            span_days = (trades[-1]["timestamp"] - trades[0]["timestamp"]) / 86400
        else:
            span_days = 0
        bid_fills = sum(1 for f in fills if f["side"] == "BUY")
        ask_fills = sum(1 for f in fills if f["side"] == "SELL")
        fill_rate = len(fills) / total_orders * 100 if total_orders > 0 else 0

        # Adverse selection
        adverse = 0
        for f in fills:
            move = f["mid_after_60s"] - f["mid_at_fill"]
            if (f["side"] == "BUY" and move < 0) or (f["side"] == "SELL" and move > 0):
                adverse += 1
        adverse_pct = adverse / len(fills) * 100 if fills else 0

        logger.info(f"  span: {span_days:.1f}d | orders: {total_orders} | fills: {len(fills)} ({fill_rate:.1f}%)")
        logger.info(f"  bid fills: {bid_fills} | ask fills: {ask_fills}")
        logger.info(f"  adverse: {adverse}/{len(fills)} ({adverse_pct:.0f}%)")
        logger.info(f"  queue depth: {avg_queue:.0f} | fill prob: {min_size/(min_size+avg_queue):.1%}")

        all_results.append({
            "question": question,
            "span_days": span_days,
            "trades": len(trades),
            "orders": total_orders,
            "fills": len(fills),
            "fill_rate": fill_rate,
            "adverse_pct": adverse_pct,
            "avg_queue": avg_queue,
        })

    http.close()
    client.close()
    conn.close()

    # Print calibration from historical replay
    logger.info("\n" + "=" * 60)
    logger.info("HISTORICAL REPLAY CALIBRATION")
    logger.info("=" * 60)
    cal = get_calibration(hist_db_path)
    for k, v in cal.items():
        if k != "_summary":
            logger.info(f"  {k}: {v}")
    logger.info(f"\n  SUMMARY: {cal.get('_summary', '')}")
    logger.info(f"\nFull stats: python3 -m mantis.paper_trader --stats-hist")

    return cal


# ── Main Runner ──

def run_paper_trader(top_n: int = 5, interval: int = 30):
    """Main paper trading loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    import os
    os.chdir(str(_MODULE_DIR))

    config = load_config("config.yaml")
    client = PolymarketClient()
    conn = init_db()

    pt = PaperTrader(config, client, conn)

    # Select markets (reuse collector's logic)
    from .collector import select_markets
    logger.info(f"Selecting top {top_n} markets...")
    markets = select_markets(client, config, top_n)
    logger.info(f"Paper trading {len(markets)} markets:")

    for i, m in enumerate(markets):
        logger.info(
            f"  #{i+1}: ${m['daily_rate']}/d | rpc={m['reward_per_capital']:.2f} | "
            f"{m['question'][:50]}"
        )
        pt.add_market(m)

    # Run loop
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
        logger.info("Shutting down...")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cycle = 0
    logger.info(f"Starting paper trader: {len(markets)} markets, {interval}s interval")
    logger.info(f"Data: {DB_PATH}")
    logger.info("Press Ctrl+C to stop. Run with --stats to see results.")

    try:
        while running:
            cycle += 1
            t0 = time.time()
            pt.tick()
            elapsed = time.time() - t0

            if cycle % 20 == 1:
                # Print summary every 20 cycles
                total_fills = sum(
                    t.total_bid_fills + t.total_ask_fills
                    for t in pt.trackers.values()
                )
                total_trades = sum(t.total_trades_seen for t in pt.trackers.values())
                logger.info(
                    f"Cycle {cycle} ({elapsed:.1f}s) | "
                    f"Fills: {total_fills} | Trades seen: {total_trades}"
                )

            wait = max(0, interval - elapsed)
            if wait > 0 and running:
                time.sleep(wait)
    finally:
        client.close()
        conn.close()
        logger.info("Paper trader stopped.")
        logger.info(f"Run: python3 -m mantis.paper_trader --stats")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mantis Paper Trader")
    parser.add_argument("--top", type=int, default=5, help="Number of markets")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between ticks")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--stats-hist", action="store_true", help="Show historical replay stats")
    parser.add_argument("--calibrate", action="store_true", help="Output calibration params")
    parser.add_argument("--replay", action="store_true", help="Historical replay mode (instant)")
    args = parser.parse_args()

    if args.stats:
        print(get_stats())
    elif args.stats_hist:
        print(get_stats(_MODULE_DIR / "data" / "paper_trades_hist.db"))
    elif args.calibrate:
        import json as _json
        params = get_calibration()
        print(_json.dumps(params, indent=2))
    elif args.replay:
        run_historical_replay(top_n=args.top)
    else:
        run_paper_trader(top_n=args.top, interval=args.interval)
