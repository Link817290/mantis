"""SQLite database layer for persistent state."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("mantis.db")

# Use absolute path relative to module location
_MODULE_DIR = Path(__file__).parent.parent.resolve()
DB_PATH = _MODULE_DIR / "data" / "mantis.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS active_markets (
    condition_id TEXT PRIMARY KEY,
    question TEXT,
    token_id_yes TEXT,
    token_id_no TEXT,
    allocated_capital REAL,
    reward_rate REAL,
    max_spread REAL,
    added_at REAL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    condition_id TEXT,
    token_id TEXT,
    side TEXT,
    price REAL,
    size REAL,
    status TEXT DEFAULT 'open',
    created_at REAL,
    cancelled_at REAL
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT,
    condition_id TEXT,
    token_id TEXT,
    side TEXT,
    price REAL,
    size REAL,
    filled_at REAL
);

CREATE TABLE IF NOT EXISTS positions (
    token_id TEXT PRIMARY KEY,
    condition_id TEXT,
    outcome TEXT,
    size REAL DEFAULT 0,
    avg_cost REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS pnl_daily (
    date TEXT PRIMARY KEY,
    starting_value REAL,
    ending_value REAL,
    spread_income REAL DEFAULT 0,
    reward_income REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    gas_cost REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    details TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS cfr_state (
    strategy_key TEXT PRIMARY KEY,
    cumulative_regret REAL DEFAULT 0,
    cumulative_strategy REAL DEFAULT 0,
    rounds INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT,
    midpoint REAL,
    spread REAL,
    timestamp REAL
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_orders_condition ON orders(condition_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_fills_condition ON fills(condition_id);
CREATE INDEX IF NOT EXISTS idx_fills_time ON fills(filled_at);
CREATE INDEX IF NOT EXISTS idx_positions_condition ON positions(condition_id);
CREATE INDEX IF NOT EXISTS idx_price_history_condition ON price_history(condition_id);
CREATE INDEX IF NOT EXISTS idx_price_history_time ON price_history(timestamp);

-- Slippage tracking for model iteration
CREATE TABLE IF NOT EXISTS slippage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT,
    order_id TEXT,
    side TEXT,
    expected_price REAL,
    actual_price REAL,
    slippage_cents REAL,
    size REAL,
    orderbook_age_sec REAL,
    timestamp REAL
);

-- Price trajectory after fills (adverse selection analysis)
CREATE TABLE IF NOT EXISTS fill_trajectory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fill_id TEXT,
    condition_id TEXT,
    fill_side TEXT,
    fill_price REAL,
    mid_at_fill REAL,
    mid_after_10s REAL,
    mid_after_30s REAL,
    mid_after_60s REAL,
    mid_after_300s REAL,
    max_adverse_move REAL,
    realized_pnl REAL,
    timestamp REAL
);

CREATE INDEX IF NOT EXISTS idx_slippage_condition ON slippage_log(condition_id);
CREATE INDEX IF NOT EXISTS idx_trajectory_condition ON fill_trajectory(condition_id);
"""


class Database:
    def __init__(self, path: Path | str = DB_PATH):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._maybe_commit()
        self._batch_mode = False
        self._pending_commits = 0

    def close(self):
        if self._pending_commits > 0:
            self._maybe_commit()
        self._conn.close()

    def begin_batch(self):
        """Start batch mode - delays commits until end_batch()."""
        self._batch_mode = True
        self._pending_commits = 0

    def end_batch(self):
        """End batch mode and commit all pending changes."""
        if self._pending_commits > 0:
            self._maybe_commit()
        self._batch_mode = False
        self._pending_commits = 0

    def _maybe_commit(self):
        """Commit unless in batch mode."""
        if self._batch_mode:
            self._pending_commits += 1
            # Auto-commit every 100 operations to prevent huge transactions
            if self._pending_commits >= 100:
                self._maybe_commit()
                self._pending_commits = 0
        else:
            self._maybe_commit()

    # ── Active Markets ──

    def set_active_market(
        self, condition_id: str, question: str,
        token_yes: str, token_no: str,
        capital: float, reward_rate: float, max_spread: float,
    ):
        self._conn.execute(
            """INSERT OR REPLACE INTO active_markets
               (condition_id, question, token_id_yes, token_id_no,
                allocated_capital, reward_rate, max_spread, added_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (condition_id, question, token_yes, token_no,
             capital, reward_rate, max_spread, time.time()),
        )
        self._maybe_commit()

    def get_active_markets(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM active_markets").fetchall()
        return [dict(r) for r in rows]

    def remove_active_market(self, condition_id: str):
        self._conn.execute(
            "DELETE FROM active_markets WHERE condition_id = ?", (condition_id,),
        )
        self._maybe_commit()

    # ── Orders ──

    def record_order(
        self, order_id: str, condition_id: str, token_id: str,
        side: str, price: float, size: float,
    ):
        self._conn.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, condition_id, token_id, side, price, size, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'open', ?)""",
            (order_id, condition_id, token_id, side, price, size, time.time()),
        )
        self._maybe_commit()

    def mark_order_cancelled(self, order_id: str):
        self._conn.execute(
            "UPDATE orders SET status='cancelled', cancelled_at=? WHERE order_id=?",
            (time.time(), order_id),
        )
        self._maybe_commit()

    def get_open_orders(self, condition_id: str = "") -> list[dict]:
        if condition_id:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE status='open' AND condition_id=?",
                (condition_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE status='open'",
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Fills ──

    def record_fill(
        self, order_id: str, condition_id: str, token_id: str,
        side: str, price: float, size: float,
    ):
        self._conn.execute(
            """INSERT INTO fills (order_id, condition_id, token_id, side, price, size, filled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (order_id, condition_id, token_id, side, price, size, time.time()),
        )
        self._maybe_commit()

    def get_recent_fills(self, condition_id: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM fills WHERE condition_id=? ORDER BY filled_at DESC LIMIT ?",
            (condition_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Positions ──

    def update_position(
        self, token_id: str, condition_id: str, outcome: str,
        size: float, avg_cost: float,
    ):
        self._conn.execute(
            """INSERT OR REPLACE INTO positions
               (token_id, condition_id, outcome, size, avg_cost)
               VALUES (?, ?, ?, ?, ?)""",
            (token_id, condition_id, outcome, size, avg_cost),
        )
        self._maybe_commit()

    def get_positions(self, condition_id: str = "") -> list[dict]:
        if condition_id:
            rows = self._conn.execute(
                "SELECT * FROM positions WHERE condition_id=? AND size > 0",
                (condition_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM positions WHERE size > 0",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_positions(self) -> list[dict]:
        """Get all positions across all markets (for capital accounting)."""
        return self.get_positions()

    def clear_all_positions(self):
        """Clear all positions (used before chain sync)."""
        self._conn.execute("DELETE FROM positions")
        self._maybe_commit()

    # ── PnL ──

    def record_daily_pnl(self, date: str, **kwargs):
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * (len(kwargs) + 1))
        sets = ", ".join(f"{k}=?" for k in kwargs.keys())
        vals = list(kwargs.values())
        self._conn.execute(
            f"""INSERT INTO pnl_daily (date, {cols}) VALUES (?, {', '.join(['?']*len(kwargs))})
                ON CONFLICT(date) DO UPDATE SET {sets}""",
            [date] + vals + vals,
        )
        self._maybe_commit()

    # ── Risk Events ──

    def record_risk_event(self, event_type: str, details: str):
        self._conn.execute(
            "INSERT INTO risk_events (event_type, details, created_at) VALUES (?, ?, ?)",
            (event_type, details, time.time()),
        )
        self._maybe_commit()

    # ── CFR State ──

    def get_cfr_state(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM cfr_state").fetchall()
        return [dict(r) for r in rows]

    def update_cfr_state(self, key: str, regret: float, strategy: float, rounds: int):
        self._conn.execute(
            """INSERT OR REPLACE INTO cfr_state
               (strategy_key, cumulative_regret, cumulative_strategy, rounds)
               VALUES (?, ?, ?, ?)""",
            (key, regret, strategy, rounds),
        )
        self._maybe_commit()

    # ── Price History ──

    def record_price(self, condition_id: str, midpoint: float, spread: float):
        self._conn.execute(
            "INSERT INTO price_history (condition_id, midpoint, spread, timestamp) VALUES (?, ?, ?, ?)",
            (condition_id, midpoint, spread, time.time()),
        )
        self._maybe_commit()

    def get_recent_prices(self, condition_id: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM price_history WHERE condition_id=? ORDER BY timestamp DESC LIMIT ?",
            (condition_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Order Status Updates ──

    def mark_order_filled(self, order_id: str):
        """Mark order as filled (called when fill is confirmed)."""
        self._conn.execute(
            "UPDATE orders SET status='filled' WHERE order_id=?",
            (order_id,),
        )
        self._maybe_commit()

    # ── Slippage Tracking ──

    def record_slippage(
        self, condition_id: str, order_id: str, side: str,
        expected_price: float, actual_price: float, slippage_cents: float,
        size: float, orderbook_age_sec: float,
    ):
        """Record slippage for model analysis."""
        self._conn.execute(
            """INSERT INTO slippage_log
               (condition_id, order_id, side, expected_price, actual_price,
                slippage_cents, size, orderbook_age_sec, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (condition_id, order_id, side, expected_price, actual_price,
             slippage_cents, size, orderbook_age_sec, time.time()),
        )
        self._maybe_commit()

    def get_slippage_stats(self, condition_id: str = "", days: int = 7) -> dict:
        """Get slippage statistics for analysis."""
        cutoff = time.time() - days * 86400
        if condition_id:
            rows = self._conn.execute(
                """SELECT AVG(slippage_cents) as avg_slip,
                          MAX(slippage_cents) as max_slip,
                          COUNT(*) as n_fills
                   FROM slippage_log
                   WHERE condition_id=? AND timestamp > ?""",
                (condition_id, cutoff),
            ).fetchone()
        else:
            rows = self._conn.execute(
                """SELECT AVG(slippage_cents) as avg_slip,
                          MAX(slippage_cents) as max_slip,
                          COUNT(*) as n_fills
                   FROM slippage_log WHERE timestamp > ?""",
                (cutoff,),
            ).fetchone()
        return dict(rows) if rows else {"avg_slip": 0, "max_slip": 0, "n_fills": 0}

    # ── Price Trajectory ──

    def record_trajectory(
        self, fill_id: str, condition_id: str, fill_side: str,
        fill_price: float, mid_at_fill: float,
        mid_after_10s: float, mid_after_30s: float,
        mid_after_60s: float, mid_after_300s: float,
        max_adverse_move: float, realized_pnl: float,
    ):
        """Record price trajectory after fill for adverse selection analysis."""
        self._conn.execute(
            """INSERT INTO fill_trajectory
               (fill_id, condition_id, fill_side, fill_price, mid_at_fill,
                mid_after_10s, mid_after_30s, mid_after_60s, mid_after_300s,
                max_adverse_move, realized_pnl, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fill_id, condition_id, fill_side, fill_price, mid_at_fill,
             mid_after_10s, mid_after_30s, mid_after_60s, mid_after_300s,
             max_adverse_move, realized_pnl, time.time()),
        )
        self._maybe_commit()

    def get_adverse_selection_stats(self, condition_id: str = "", days: int = 7) -> dict:
        """Get adverse selection statistics for model tuning."""
        cutoff = time.time() - days * 86400
        if condition_id:
            rows = self._conn.execute(
                """SELECT AVG(max_adverse_move) as avg_adverse,
                          AVG(realized_pnl) as avg_pnl,
                          COUNT(*) as n_fills,
                          SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as n_losing
                   FROM fill_trajectory
                   WHERE condition_id=? AND timestamp > ?""",
                (condition_id, cutoff),
            ).fetchone()
        else:
            rows = self._conn.execute(
                """SELECT AVG(max_adverse_move) as avg_adverse,
                          AVG(realized_pnl) as avg_pnl,
                          COUNT(*) as n_fills,
                          SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as n_losing
                   FROM fill_trajectory WHERE timestamp > ?""",
                (cutoff,),
            ).fetchone()
        return dict(rows) if rows else {"avg_adverse": 0, "avg_pnl": 0, "n_fills": 0, "n_losing": 0}

    # ── Data Retention / Cleanup ──

    def cleanup_old_data(self, days: int = 30):
        """Remove data older than N days to prevent DB bloat."""
        cutoff = time.time() - days * 86400
        tables_to_clean = [
            ("price_history", "timestamp"),
            ("slippage_log", "timestamp"),
            ("fill_trajectory", "timestamp"),
            ("fills", "filled_at"),
        ]
        total_deleted = 0
        for table, time_col in tables_to_clean:
            cursor = self._conn.execute(
                f"DELETE FROM {table} WHERE {time_col} < ?", (cutoff,)
            )
            total_deleted += cursor.rowcount

        # Clean up old cancelled orders (keep recent 7 days)
        order_cutoff = time.time() - 7 * 86400
        cursor = self._conn.execute(
            "DELETE FROM orders WHERE status IN ('cancelled', 'filled') AND created_at < ?",
            (order_cutoff,),
        )
        total_deleted += cursor.rowcount

        self._maybe_commit()
        logger.info(f"Cleaned up {total_deleted} old records")
        return total_deleted

    def vacuum(self):
        """Reclaim disk space after cleanup."""
        self._conn.execute("VACUUM")

    # ── Analytics Helpers ──

    def get_fill_summary(self, days: int = 7) -> dict:
        """Get fill summary for performance analysis."""
        cutoff = time.time() - days * 86400
        rows = self._conn.execute(
            """SELECT side,
                      COUNT(*) as n_fills,
                      SUM(size) as total_size,
                      AVG(price) as avg_price
               FROM fills WHERE filled_at > ?
               GROUP BY side""",
            (cutoff,),
        ).fetchall()
        return {r["side"]: dict(r) for r in rows}

    def get_pnl_history(self, days: int = 30) -> list[dict]:
        """Get daily PnL history."""
        rows = self._conn.execute(
            """SELECT * FROM pnl_daily
               ORDER BY date DESC LIMIT ?""",
            (days,),
        ).fetchall()
        return [dict(r) for r in rows]
