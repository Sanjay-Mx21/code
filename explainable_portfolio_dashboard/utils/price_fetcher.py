# utils/price_fetcher.py
from typing import List
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_live_prices(tickers: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch OHLCV daily prices for the given tickers between start and end (inclusive).
    Returns a DataFrame with columns: ['ticker','date','open','high','low','close','adj_close','volume'].
    """
    if not tickers:
        return pd.DataFrame(columns=['ticker','date','open','high','low','close','adj_close','volume'])

    # yfinance accepts tickers as string list or space-separated
    tick_str = " ".join(tickers)
    # period download - use yf.download which returns multi-index (ticker, date) when multiple
    df = yf.download(tickers=tick_str, start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d", progress=False, threads=True)

    # If single ticker, yfinance returns columns without ticker-level; normalize
    if isinstance(df.columns, pd.MultiIndex):
        # df: columns like ('Open','AAPL') or ( 'AAPL', 'Open' ) depending on version — normalize:
        # transpose to (date, ticker, col)
        df = df.stack(level=1).reset_index().rename(columns={"level_1":"ticker"})
    else:
        # single ticker: make ticker column explicit
        df = df.reset_index()
        # try to detect ticker from input
        if len(tickers) == 1:
            df["ticker"] = tickers[0].upper()
        else:
            # fallback: set ticker unknown
            df["ticker"] = None

    # Normalize column names
    df = df.rename(columns={
        "Date": "date", "date": "date",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Adj Close": "adj_close",
        "Volume": "volume", "ticker": "ticker"
    })

    # Keep only needed columns and ensure types
    cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper()

    return df
