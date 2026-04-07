"""Polymarket API client for market data and trading."""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .strategy_log import slog
from .types import (
    Market, MarketRewards, Orderbook, OrderLevel, Token,
)

logger = logging.getLogger("mantis.client")

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mantis/0.1",
}


class PolymarketClient:
    """Read-only Polymarket API client for market data."""

    def __init__(self):
        self._http = httpx.Client(headers=HEADERS, timeout=15)

    def close(self):
        self._http.close()

    # ── Positions ──

    def fetch_positions(self, proxy_address: str) -> list[dict]:
        """Fetch real on-chain positions from Polymarket data API.

        Returns list of dicts with keys: asset, conditionId, outcome, size,
        avgPrice, curPrice, cashPnl, title, etc.
        """
        url = f"https://data-api.polymarket.com/positions"
        resp = self._http.get(url, params={"user": proxy_address})
        resp.raise_for_status()
        return [p for p in resp.json() if p.get("size", 0) > 0]

    # ── Market Discovery ──

    def fetch_sampling_markets(self, cursor: str = "") -> tuple[list[Market], str]:
        """Fetch reward-eligible markets from CLOB sampling endpoint.

        Returns (markets, next_cursor).
        """
        url = f"{CLOB_BASE}/sampling-markets"
        params: dict[str, Any] = {}
        if cursor:
            params["next_cursor"] = cursor

        resp = self._http.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        markets = []
        for m in data.get("data", []):
            rewards_raw = m.get("rewards", {})
            rates = rewards_raw.get("rates", [{}])
            daily_rate = rates[0].get("rewards_daily_rate", 0) if rates else 0

            tokens = []
            for t in m.get("tokens", []):
                tokens.append(Token(
                    token_id=t["token_id"],
                    outcome=t["outcome"],
                    price=t.get("price", 0),
                ))

            market = Market(
                condition_id=m["condition_id"],
                question=m.get("question", ""),
                tokens=tokens,
                rewards=MarketRewards(
                    daily_rate=daily_rate,
                    min_size=rewards_raw.get("min_size", 20),
                    max_spread=rewards_raw.get("max_spread", 3.5),
                ),
                end_date=m.get("end_date_iso", ""),
                active=m.get("active", True),
            )
            markets.append(market)

        next_cursor = data.get("next_cursor", "")
        return markets, next_cursor

    def fetch_all_sampling_markets(self) -> list[Market]:
        """Fetch all pages of sampling markets."""
        all_markets: list[Market] = []
        cursor = ""
        while True:
            markets, next_cursor = self.fetch_sampling_markets(cursor)
            all_markets.extend(markets)
            if not next_cursor or next_cursor == "LTE=" or not markets:
                break
            cursor = next_cursor
        logger.info(f"Fetched {len(all_markets)} sampling markets")
        return all_markets

    # ── Quick Price (Midpoint API - 更轻量) ──

    def fetch_midpoint(self, token_id: str) -> float | None:
        """快速获取中间价（比获取整个订单簿更轻量）。"""
        url = f"{CLOB_BASE}/midpoint"
        try:
            resp = self._http.get(url, params={"token_id": token_id}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", data.get("mid_price", 0)))
        except Exception as e:
            logger.debug(f"Midpoint fetch failed: {e}")
            return None

    def fetch_spread(self, token_id: str) -> dict | None:
        """获取最优买卖价和价差。"""
        url = f"{CLOB_BASE}/spread"
        try:
            resp = self._http.get(url, params={"token_id": token_id}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return {
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "spread": float(data.get("spread", 0)),
            }
        except Exception as e:
            logger.debug(f"Spread fetch failed: {e}")
            return None

    # ── Orderbook ──

    def fetch_orderbook(self, token_id: str) -> Orderbook:
        """Fetch orderbook for a token."""
        url = f"{CLOB_BASE}/book"
        t0 = time.time()
        resp = self._http.get(url, params={"token_id": token_id})
        resp.raise_for_status()
        slog.api_call(method="GET", endpoint="book",
                      latency_ms=(time.time() - t0) * 1000,
                      token_id=token_id[:16])
        data = resp.json()

        bids = [
            OrderLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: -x.price)
        asks.sort(key=lambda x: x.price)

        return Orderbook(bids=bids, asks=asks, timestamp=time.time())

    # ── Price History (from gamma API) ──

    def fetch_market_by_condition(self, condition_id: str) -> dict | None:
        """Fetch market details from gamma API."""
        url = f"{GAMMA_BASE}/markets"
        resp = self._http.get(url, params={"condition_id": condition_id, "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else None


    def fetch_trades(self, condition_id: str, maker: str = "", limit: int = 50) -> list[dict]:
        """Fetch recent trades for a market.

        Returns list of trade dicts with keys: id, price, size, side, timestamp, etc.
        """
        url = f"{CLOB_BASE}/trades"
        params: dict[str, Any] = {"condition_id": condition_id, "limit": limit}
        if maker:
            params["maker"] = maker
        try:
            resp = self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json() or []
        except Exception as e:
            logger.debug(f"Failed to fetch trades: {e}")
            return []


class PolymarketTrader:
    """Trading client using py-clob-client for order placement.

    This is a separate class because it requires wallet credentials.
    """

    def __init__(self, private_key: str, funder: str = "",
                 chain_id: int = 137, signature_type: int = 1):
        self._private_key = private_key
        self._funder = funder
        self._chain_id = chain_id
        self._signature_type = signature_type
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from py_clob_client.client import ClobClient
            kwargs: dict = {
                "host": CLOB_BASE,
                "key": self._private_key,
                "chain_id": self._chain_id,
                "signature_type": self._signature_type,
            }
            if self._funder:
                kwargs["funder"] = self._funder
            self._client = ClobClient(**kwargs)
            # Derive API credentials
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
            logger.info("Trading client initialized")
        except ImportError:
            raise RuntimeError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )

    def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> dict:
        """Place a limit order. Returns order response dict."""
        self._ensure_client()
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL

        order_side = BUY if side == "BUY" else SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=order_side,
        )
        order = self._client.create_order(order_args)
        t0 = time.time()
        result = self._client.post_order(order)
        slog.api_call(method="POST", endpoint="post_order",
                      latency_ms=(time.time() - t0) * 1000,
                      side=side, price=price, size=size)
        logger.info(f"Order placed: {side} {size}@{price} → {result}")
        return result

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an existing order."""
        self._ensure_client()
        t0 = time.time()
        result = self._client.cancel(order_id)
        slog.api_call(method="DELETE", endpoint="cancel_order",
                      latency_ms=(time.time() - t0) * 1000,
                      order_id=order_id)
        logger.info(f"Order cancelled: {order_id}")
        return result

    def cancel_all(self) -> dict:
        """Cancel all open orders."""
        self._ensure_client()
        result = self._client.cancel_all()
        logger.info("All orders cancelled")
        return result

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        self._ensure_client()
        return self._client.get_orders() or []

    def get_trades(self) -> list[dict]:
        """Get recent trades (authenticated)."""
        self._ensure_client()
        return self._client.get_trades() or []

    # ── Heartbeat (防止订单被自动取消) ──

    def send_heartbeat(self) -> bool:
        """发送心跳保持会话活跃。

        重要：如果不定期发送心跳，所有订单会被自动取消！
        建议每 30 秒发送一次。
        """
        self._ensure_client()
        try:
            # py-clob-client 可能没有直接的 heartbeat 方法
            # 使用底层 HTTP 调用
            import hmac
            import hashlib
            import base64

            ts = str(int(time.time()))
            api_key = self._client.creds.api_key
            api_secret = self._client.creds.api_secret
            api_passphrase = self._client.creds.api_passphrase

            # 签名
            message = ts + "POST" + "/heartbeats"
            signature = base64.b64encode(
                hmac.new(
                    base64.b64decode(api_secret),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode()

            headers = {
                "POLY_API_KEY": api_key,
                "POLY_ADDRESS": self._client.get_address(),
                "POLY_SIGNATURE": signature,
                "POLY_PASSPHRASE": api_passphrase,
                "POLY_TIMESTAMP": ts,
            }

            resp = httpx.post(
                f"{CLOB_BASE}/heartbeats",
                headers=headers,
                timeout=5,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
            return False

    # ── 批量操作 ──

    def create_multiple_orders(self, orders: list[dict]) -> list[dict]:
        """批量下单（最多15个）。

        orders: list of {token_id, side, price, size}
        Returns: list of order responses
        """
        self._ensure_client()
        if len(orders) > 15:
            raise ValueError("Maximum 15 orders per batch")

        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY, SELL

        built_orders = []
        for o in orders:
            order_side = BUY if o["side"] == "BUY" else SELL
            order_args = OrderArgs(
                token_id=o["token_id"],
                price=o["price"],
                size=o["size"],
                side=order_side,
            )
            built_orders.append(self._client.create_order(order_args))

        t0 = time.time()
        # py-clob-client 的批量接口
        try:
            result = self._client.post_orders(built_orders)
        except AttributeError:
            # 如果没有 post_orders 方法，回退到单个下单
            result = [self._client.post_order(o) for o in built_orders]

        slog.api_call(
            method="POST", endpoint="post_orders",
            latency_ms=(time.time() - t0) * 1000,
            n_orders=len(orders),
        )
        logger.info(f"Batch order placed: {len(orders)} orders")
        return result

    def cancel_orders(self, order_ids: list[str]) -> dict:
        """批量取消订单（最多3000个）。"""
        self._ensure_client()
        if not order_ids:
            return {"cancelled": 0}

        t0 = time.time()
        try:
            result = self._client.cancel_orders(order_ids)
        except AttributeError:
            # 回退到逐个取消
            for oid in order_ids:
                try:
                    self._client.cancel(oid)
                except Exception:
                    pass
            result = {"cancelled": len(order_ids)}

        slog.api_call(
            method="DELETE", endpoint="cancel_orders",
            latency_ms=(time.time() - t0) * 1000,
            n_orders=len(order_ids),
        )
        logger.info(f"Batch cancel: {len(order_ids)} orders")
        return result

    def cancel_market_orders(self, market_id: str = "", asset_id: str = "") -> dict:
        """取消特定市场的所有订单。

        market_id: condition_id
        asset_id: token_id
        """
        self._ensure_client()
        t0 = time.time()
        try:
            result = self._client.cancel_market_orders(
                market=market_id,
                asset_id=asset_id,
            )
        except AttributeError:
            # 回退：获取所有订单然后过滤取消
            all_orders = self.get_open_orders()
            to_cancel = [
                o.get("id", o.get("orderID", ""))
                for o in all_orders
                if (not market_id or o.get("market") == market_id) and
                   (not asset_id or o.get("asset_id") == asset_id)
            ]
            for oid in to_cancel:
                try:
                    self._client.cancel(oid)
                except Exception:
                    pass
            result = {"cancelled": len(to_cancel)}

        slog.api_call(
            method="DELETE", endpoint="cancel_market_orders",
            latency_ms=(time.time() - t0) * 1000,
            market_id=market_id[:16] if market_id else "",
        )
        return result

    def get_order(self, order_id: str) -> dict | None:
        """获取单个订单详情。"""
        self._ensure_client()
        try:
            return self._client.get_order(order_id)
        except Exception:
            return None

    def get_orders_for_market(self, market_id: str) -> list[dict]:
        """获取特定市场的所有挂单。"""
        self._ensure_client()
        try:
            return self._client.get_orders(market=market_id) or []
        except Exception:
            # 回退：获取所有订单然后过滤
            all_orders = self.get_open_orders()
            return [o for o in all_orders if o.get("market") == market_id]
