"""Backtest calibration tool - combines data from paper_trader and collector.

Generates calibrated parameters for realistic_backtest.py based on:
1. Paper trading data (fill probability, adverse selection, queue time)
2. Collector data (orderbook dynamics, VPIN, Q competition)

Usage:
    python scripts/calibrate_backtest.py              # auto-calibrate
    python scripts/calibrate_backtest.py --report     # detailed report
    python scripts/calibrate_backtest.py --export     # export as Python dict
    python scripts/calibrate_backtest.py --update     # update realistic_backtest.py
"""
import sys
from pathlib import Path

# Cross-platform path setup
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

import json
import sqlite3
from datetime import datetime

# Data paths relative to project root
PAPER_DB = _SCRIPT_DIR / "data" / "paper_trades.db"
PAPER_HIST_DB = _SCRIPT_DIR / "data" / "paper_trades_hist.db"
COLLECTOR_DB = _SCRIPT_DIR / "data" / "snapshots.db"


def load_paper_trader_calibration(db_path: Path) -> dict:
    """Load calibration from paper_trader database."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    result = {}

    # Fill probability
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

    # Queue time
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
    except sqlite3.OperationalError:
        pass

    # Adverse selection
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
        if moves:
            sorted_moves = sorted(moves)
            result["adverse_move_p50"] = sorted_moves[len(sorted_moves) // 2]
            result["adverse_move_p90"] = sorted_moves[int(len(sorted_moves) * 0.9)]

    # Toxic flow
    try:
        toxic_fills = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE status='filled' AND is_toxic=1"
        ).fetchone()[0]
        result["toxic_fill_rate"] = toxic_fills / filled if filled > 0 else 0

        # VPIN correlation
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

    conn.close()
    return result


def load_collector_calibration(db_path: Path) -> dict:
    """Load calibration from collector database."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    result = {}

    # Queue depth
    row = conn.execute("""
        SELECT AVG(bid_depth_5), AVG(ask_depth_5), AVG(total_q)
        FROM snapshots
    """).fetchone()
    if row:
        result["avg_bid_depth_5"] = row[0] or 0
        result["avg_ask_depth_5"] = row[1] or 0
        result["avg_total_q"] = row[2] or 0

    # VPIN
    try:
        row = conn.execute("""
            SELECT AVG(vpin), AVG(trade_imbalance) FROM snapshots WHERE vpin > 0
        """).fetchone()
        if row and row[0]:
            result["avg_vpin"] = row[0]
            result["avg_trade_imbalance"] = row[1] or 0

        # High VPIN frequency
        high = conn.execute("SELECT COUNT(*) FROM snapshots WHERE vpin > 0.35").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM snapshots WHERE vpin > 0").fetchone()[0]
        if total > 0:
            result["high_vpin_frequency"] = high / total
    except sqlite3.OperationalError:
        pass

    # Trade frequency
    try:
        row = conn.execute("""
            SELECT COUNT(*), MIN(ts_unix), MAX(ts_unix) FROM trades
        """).fetchone()
        if row and row[0] and row[1] and row[2]:
            hours = (row[2] - row[1]) / 3600
            if hours > 0:
                result["avg_trades_per_hour"] = row[0] / hours
    except sqlite3.OperationalError:
        pass

    # Orderbook dynamics
    try:
        row = conn.execute("""
            SELECT AVG(bid_adds), AVG(bid_removes), AVG(ask_adds), AVG(ask_removes)
            FROM snapshots WHERE bid_adds > 0 OR ask_adds > 0
        """).fetchone()
        if row and row[0]:
            result["avg_bid_queue_adds"] = row[0]
            result["avg_bid_queue_removes"] = row[1]
            result["avg_ask_queue_adds"] = row[2]
            result["avg_ask_queue_removes"] = row[3]
    except sqlite3.OperationalError:
        pass

    # Q competition
    try:
        row = conn.execute("""
            SELECT AVG(depth_at_best_bid), AVG(depth_at_best_ask), AVG(n_makers_at_best)
            FROM q_competition
        """).fetchone()
        if row and row[0]:
            result["avg_depth_at_best_bid"] = row[0]
            result["avg_depth_at_best_ask"] = row[1]
            result["avg_makers_at_best"] = row[2]
    except sqlite3.OperationalError:
        pass

    conn.close()
    return result


def merge_calibrations(paper: dict, collector: dict) -> dict:
    """Merge and reconcile calibrations from different sources."""
    result = {}

    # Fill probability: prefer paper trader data (more accurate)
    result["fill_prob"] = paper.get("pro_rata_fill_prob", 0.278)

    # Queue depth: average from both sources
    paper_queue = paper.get("avg_queue_depth", 0)
    collector_queue = (collector.get("avg_depth_at_best_bid", 0) +
                       collector.get("avg_depth_at_best_ask", 0)) / 2
    if paper_queue > 0 and collector_queue > 0:
        result["queue_depth"] = (paper_queue + collector_queue) / 2
    elif paper_queue > 0:
        result["queue_depth"] = paper_queue
    elif collector_queue > 0:
        result["queue_depth"] = collector_queue
    else:
        result["queue_depth"] = 130  # default

    # Queue time
    result["queue_time_avg"] = paper.get("avg_queue_time_sec", 120)
    result["queue_time_p50"] = paper.get("queue_time_p50", 90)
    result["queue_time_p90"] = paper.get("queue_time_p90", 240)

    # Adverse selection
    result["adverse_rate"] = paper.get("adverse_selection_rate", 0.06)
    result["adverse_move_avg"] = paper.get("avg_adverse_move", 0.0155)
    result["adverse_move_p90"] = paper.get("adverse_move_p90", 0.03)

    # VPIN-based adverse rates
    result["adverse_rate_high_vpin"] = paper.get("adverse_rate_high_vpin", 0.18)
    result["adverse_rate_low_vpin"] = paper.get("adverse_rate_low_vpin", 0.04)

    # Toxic flow
    result["toxic_fill_rate"] = paper.get("toxic_fill_rate", 0.15)

    # Trade frequency
    result["trades_per_hour"] = collector.get("avg_trades_per_hour", 5.0)

    # Q competition
    result["total_q_avg"] = collector.get("avg_total_q", 100)

    # VPIN baseline
    result["vpin_avg"] = collector.get("avg_vpin", 0.2)
    result["high_vpin_frequency"] = collector.get("high_vpin_frequency", 0.1)

    return result


def generate_report(calibration: dict) -> str:
    """Generate detailed calibration report."""
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST CALIBRATION REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("--- Fill Probability ---")
    lines.append(f"  Calibrated fill prob:    {calibration.get('fill_prob', 0):.1%}")
    lines.append(f"  Avg queue depth:         {calibration.get('queue_depth', 0):.0f}")
    lines.append(f"  Avg queue time:          {calibration.get('queue_time_avg', 0):.0f}s")
    lines.append(f"  Queue time P50/P90:      {calibration.get('queue_time_p50', 0):.0f}s / {calibration.get('queue_time_p90', 0):.0f}s")
    lines.append("")

    lines.append("--- Adverse Selection ---")
    lines.append(f"  Overall adverse rate:    {calibration.get('adverse_rate', 0):.1%}")
    lines.append(f"  Avg adverse move:        {calibration.get('adverse_move_avg', 0)*100:.2f}c")
    lines.append(f"  P90 adverse move:        {calibration.get('adverse_move_p90', 0)*100:.2f}c")
    lines.append(f"  High VPIN adverse rate:  {calibration.get('adverse_rate_high_vpin', 0):.1%}")
    lines.append(f"  Low VPIN adverse rate:   {calibration.get('adverse_rate_low_vpin', 0):.1%}")
    lines.append("")

    lines.append("--- Toxic Flow ---")
    lines.append(f"  Toxic fill rate:         {calibration.get('toxic_fill_rate', 0):.1%}")
    lines.append(f"  Avg VPIN:                {calibration.get('vpin_avg', 0):.2f}")
    lines.append(f"  High VPIN frequency:     {calibration.get('high_vpin_frequency', 0):.1%}")
    lines.append("")

    lines.append("--- Market Activity ---")
    lines.append(f"  Trades per hour:         {calibration.get('trades_per_hour', 0):.1f}")
    lines.append(f"  Avg total Q:             {calibration.get('total_q_avg', 0):.0f}")
    lines.append("")

    lines.append("--- Recommended Config Updates ---")
    lines.append(f"  CALIBRATED_FILL_PROB = {calibration.get('fill_prob', 0.278):.3f}")
    lines.append(f"  CALIBRATED_ADVERSE_RATE = {calibration.get('adverse_rate', 0.06):.3f}")
    lines.append(f"  CALIBRATED_ADVERSE_MOVE = {calibration.get('adverse_move_avg', 0.0155):.4f}")
    lines.append(f"  CALIBRATED_QUEUE_DEPTH = {int(calibration.get('queue_depth', 130))}")
    lines.append(f"  CALIBRATED_ADVERSE_RATE_HIGH_VPIN = {calibration.get('adverse_rate_high_vpin', 0.18):.2f}")
    lines.append(f"  CALIBRATED_ADVERSE_RATE_LOW_VPIN = {calibration.get('adverse_rate_low_vpin', 0.04):.2f}")

    return "\n".join(lines)


def export_python_dict(calibration: dict) -> str:
    """Export calibration as Python code."""
    lines = []
    lines.append("# Auto-generated calibration from real data")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("CALIBRATION = {")
    for k, v in sorted(calibration.items()):
        if isinstance(v, float):
            lines.append(f'    "{k}": {v:.4f},')
        elif isinstance(v, int):
            lines.append(f'    "{k}": {v},')
        else:
            lines.append(f'    "{k}": {repr(v)},')
    lines.append("}")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest Calibration Tool")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--export", action="store_true", help="Export as Python dict")
    parser.add_argument("--json", action="store_true", help="Export as JSON")
    parser.add_argument("--use-hist", action="store_true", help="Use historical replay data")
    args = parser.parse_args()

    # Load data from all sources
    paper_db = PAPER_HIST_DB if args.use_hist else PAPER_DB
    paper_cal = load_paper_trader_calibration(paper_db)
    collector_cal = load_collector_calibration(COLLECTOR_DB)

    print(f"Data sources:")
    print(f"  Paper trader: {paper_db} ({'found' if paper_cal else 'not found'})")
    print(f"  Collector:    {COLLECTOR_DB} ({'found' if collector_cal else 'not found'})")
    print()

    if not paper_cal and not collector_cal:
        print("No calibration data found. Run paper_trader or collector first.")
        return

    # Merge calibrations
    calibration = merge_calibrations(paper_cal, collector_cal)

    if args.report:
        print(generate_report(calibration))
    elif args.export:
        print(export_python_dict(calibration))
    elif args.json:
        print(json.dumps(calibration, indent=2))
    else:
        # Default: short summary
        print("Calibration Summary:")
        print(f"  Fill prob: {calibration.get('fill_prob', 0):.1%}")
        print(f"  Queue depth: {calibration.get('queue_depth', 0):.0f}")
        print(f"  Adverse rate: {calibration.get('adverse_rate', 0):.1%}")
        print(f"  Toxic fill rate: {calibration.get('toxic_fill_rate', 0):.1%}")
        print()
        print("Run with --report for details, --export for Python code, --json for JSON")


if __name__ == "__main__":
    main()
