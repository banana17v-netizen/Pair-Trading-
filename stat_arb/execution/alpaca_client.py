"""Alpaca paper-trading API wrapper.

SAFETY: This module ONLY connects to the paper trading endpoint.
        The live endpoint is NEVER used here.

All credentials are loaded from .env via python-dotenv.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from dotenv import load_dotenv
from loguru import logger

# Always paper — hard-coded, not configurable
_PAPER_URL = "https://paper-api.alpaca.markets"

load_dotenv(Path(__file__).parents[2] / ".env")


def _get_credentials() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    sec = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not key or not sec:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
        )
    return key, sec


@lru_cache(maxsize=1)
def get_trading_client() -> TradingClient:
    """Return cached paper-trading TradingClient."""
    key, sec = _get_credentials()
    return TradingClient(key, sec, paper=True)


# ── Account ───────────────────────────────────────────────────────────────────

def get_account() -> dict:
    """Return account summary: status, equity, buying_power, cash."""
    client  = get_trading_client()
    account = client.get_account()
    return {
        "status":       account.status,
        "equity":       float(account.equity),
        "cash":         float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
    }


# ── Positions ─────────────────────────────────────────────────────────────────

def get_all_positions() -> dict[str, dict]:
    """Return open positions keyed by symbol.

    Returns
    -------
    dict[symbol → {qty, market_value, avg_entry_price, side, unrealized_pl}]
    """
    client    = get_trading_client()
    positions = client.get_all_positions()
    result    = {}
    for p in positions:
        result[p.symbol] = {
            "qty":             float(p.qty),
            "market_value":    float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "side":            p.side.value,           # "long" or "short"
            "unrealized_pl":   float(p.unrealized_pl),
        }
    return result


def close_position(symbol: str) -> dict:
    """Close the entire open position for a symbol via market order."""
    client = get_trading_client()
    resp   = client.close_position(symbol)
    logger.info(f"CLOSED position {symbol}: order_id={resp.id}")
    return {"order_id": str(resp.id), "symbol": symbol}


def close_all_positions() -> list[dict]:
    """Close ALL open positions (used for CLOSE_ALL risk action)."""
    client = get_trading_client()
    resp   = client.close_all_positions(cancel_orders=True)
    logger.warning(f"CLOSE_ALL: closed {len(resp)} positions")
    return [{"order_id": str(r.id), "symbol": r.symbol} for r in resp]


# ── Orders ────────────────────────────────────────────────────────────────────

def submit_market_order(
    symbol:   str,
    qty:      float,
    side:     str,       # "buy" or "sell"
    notional: float | None = None,
) -> dict:
    """Submit a fractional market order.

    Either qty (shares) OR notional (dollars) must be provided.
    Returns order info dict.
    """
    client = get_trading_client()

    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    if notional is not None:
        # Fractional order by dollar amount
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

    order = client.submit_order(req)
    logger.info(
        f"ORDER submitted: {side.upper()} {symbol}  "
        f"{'notional=$' + str(round(notional, 2)) if notional else 'qty=' + str(qty)}  "
        f"order_id={order.id}"
    )
    return {
        "order_id":  str(order.id),
        "symbol":    symbol,
        "side":      side,
        "notional":  notional,
        "qty":       qty,
        "status":    order.status.value,
    }


def get_open_orders() -> list[dict]:
    """Return all open/pending orders."""
    client = get_trading_client()
    req    = GetOrdersRequest(status="open")
    orders = client.get_orders(filter=req)
    return [
        {
            "order_id": str(o.id),
            "symbol":   o.symbol,
            "side":     o.side.value,
            "status":   o.status.value,
        }
        for o in orders
    ]


def cancel_all_orders() -> int:
    """Cancel all open orders. Returns count cancelled."""
    client = get_trading_client()
    result = client.cancel_orders()
    n      = len(result)
    if n:
        logger.info(f"Cancelled {n} open orders")
    return n


# ── Market data (current prices) ─────────────────────────────────────────────

def get_latest_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch latest trade prices for a list of symbols.

    Uses alpaca-py data client for market data.
    Falls back to yfinance if symbol not found.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        key, sec    = _get_credentials()
        data_client = StockHistoricalDataClient(key, sec)
        req         = StockLatestTradeRequest(symbol_or_symbols=symbols)
        trades      = data_client.get_stock_latest_trade(req)

        prices = {}
        for sym in symbols:
            if sym in trades:
                prices[sym] = float(trades[sym].price)
        return prices
    except Exception as exc:
        logger.warning(f"Alpaca latest prices failed ({exc}), falling back to yfinance")
        return _get_prices_yfinance(symbols)


def _get_prices_yfinance(symbols: list[str]) -> dict[str, float]:
    """Fallback price fetcher using yfinance."""
    import yfinance as yf
    prices = {}
    for sym in symbols:
        try:
            t   = yf.Ticker(sym)
            inf = t.fast_info
            prices[sym] = float(inf.last_price or inf.regularMarketPrice or 0)
        except Exception:
            pass
    return prices
