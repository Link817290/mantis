"""SQLite database layer for persistent state."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


DB_PATH = Path("mantis.db")

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
"""


class Database:
    def __init__(self, path: Path | str = DB_PATH):
        self._path = Path(path)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

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
        self._conn.commit()

    def get_active_markets(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM active_markets").fetchall()
        return [dict(r) for r in rows]

    def remove_active_market(self, condition_id: str):
        self._conn.execute(
            "DELETE FROM active_markets WHERE condition_id = ?", (condition_id,),
        )
        self._conn.commit()

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
        self._conn.commit()

    def mark_order_cancelled(self, order_id: str):
        self._conn.execute(
            "UPDATE orders SET status='cancelled', cancelled_at=? WHERE order_id=?",
            (time.time(), order_id),
        )
        self._conn.commit()

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
        self._conn.commit()

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
        self._conn.commit()

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
        self._conn.commit()

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
        self._conn.commit()

    # ── Risk Events ──

    def record_risk_event(self, event_type: str, details: str):
        self._conn.execute(
            "INSERT INTO risk_events (event_type, details, created_at) VALUES (?, ?, ?)",
            (event_type, details, time.time()),
        )
        self._conn.commit()

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
        self._conn.commit()

    # ── Price History ──

    def record_price(self, condition_id: str, midpoint: float, spread: float):
        self._conn.execute(
            "INSERT INTO price_history (condition_id, midpoint, spread, timestamp) VALUES (?, ?, ?, ?)",
            (condition_id, midpoint, spread, time.time()),
        )
        self._conn.commit()

    def get_recent_prices(self, condition_id: str, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM price_history WHERE condition_id=? ORDER BY timestamp DESC LIMIT ?",
            (condition_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
