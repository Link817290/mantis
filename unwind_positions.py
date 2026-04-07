"""Unwind existing positions by placing SELL limit orders at cost+3%."""
import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("unwind")

from dotenv import load_dotenv
load_dotenv()

import httpx
from mantis.polymarket_client import PolymarketClient, PolymarketTrader

PROXY_ADDRESS = "0x491bb5f5397e815cb010914f21f3c9b3a70e9aa2"


def fetch_positions():
    """Fetch real positions from Polymarket data API."""
    resp = httpx.get(
        f"https://data-api.polymarket.com/positions?user={PROXY_ADDRESS}",
        timeout=15,
    )
    resp.raise_for_status()
    return [p for p in resp.json() if p.get("size", 0) > 0]


def main():
    private_key = os.environ.get("MANTIS_PRIVATE_KEY", "")
    funder = os.environ.get("BROWSER_ADDRESS", "")
    dry_run = "--dry-run" in sys.argv

    if not private_key and not dry_run:
        print("ERROR: Set MANTIS_PRIVATE_KEY in .env")
        sys.exit(1)

    client = PolymarketClient()
    trader = None if dry_run else PolymarketTrader(private_key, funder=funder)

    positions = fetch_positions()
    if not positions:
        print("No positions to unwind.")
        return

    print(f"Found {len(positions)} positions to unwind\n")

    for p in positions:
        token_id = p["asset"]
        size = p["size"]
        avg_price = p.get("avgPrice", 0)
        cur_price = p.get("curPrice", 0)
        title = p.get("title", "?")[:60]
        outcome = p.get("outcome", "?")
        pnl = p.get("cashPnl", 0)

        # Get live orderbook
        ob = client.fetch_orderbook(token_id)
        best_bid = ob.bids[0].price if ob.bids else 0
        best_ask = ob.asks[0].price if ob.asks else 1

        # Target: cost + 3%, minimum at cost + 0.01
        target_price = round(avg_price * 1.03, 2)
        if target_price <= avg_price:
            target_price = round(avg_price + 0.01, 2)

        # If market already above target, sell at best_bid (instant fill)
        if best_bid >= target_price:
            sell_price = best_bid
            note = "instant fill at bid"
        else:
            sell_price = target_price
            note = "limit order, wait for fill"

        # Clamp to valid range
        sell_price = max(0.01, min(0.99, sell_price))

        print(f"{'='*55}")
        print(f"{title}")
        print(f"  {outcome} x{size}")
        print(f"  Cost: ${avg_price:.4f}  Now: ${cur_price:.4f}  P&L: ${pnl:.2f}")
        print(f"  Book: bid=${best_bid:.3f} ask=${best_ask:.3f}")
        print(f"  SELL @${sell_price:.2f} ({note})")
        print(f"  Expected return: ${size * sell_price:.2f}")

        if dry_run:
            print("  [DRY RUN] skipped")
            continue

        try:
            result = trader.create_limit_order(
                token_id=token_id,
                side="SELL",
                price=sell_price,
                size=size,
            )
            order_id = result.get("orderID", result.get("id", "unknown"))
            print(f"  ORDER PLACED! ID: {order_id}")
        except Exception as e:
            print(f"  ERROR: {e}")

    client.close()
    print(f"\n{'='*55}")
    print("Done! Monitor orders on Polymarket.")


if __name__ == "__main__":
    main()
