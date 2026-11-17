"""Precios diarios gratuitos vía yfinance, cacheados en parquet."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import CACHE

PRICE_FILE = CACHE / "prices.parquet"


def download_prices(tickers: list[str], start: str, end: str | None = None, chunk: int = 300, retries: int = 2) -> pd.DataFrame:
    """Devuelve columnas MultiIndex (campo, ticker): Close (ajustado por splits), Adj Close (total return), Volume."""
    parts = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i : i + chunk]
        for attempt in range(retries + 1):
            try:
                d = yf.download(batch, start=start, end=end, auto_adjust=False, progress=False, threads=True, group_by="column")
                if len(d):
                    parts.append(d[["Close", "Adj Close", "Volume"]])
                break
            except Exception:  # red/rate limit
                time.sleep(5 * (attempt + 1))
        print(f"  precios {min(i + chunk, len(tickers))}/{len(tickers)}", flush=True)
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def save_prices(df: pd.DataFrame) -> None:
    flat = df.copy()
    flat.columns = [f"{a}|{b}" for a, b in flat.columns]
    flat.to_parquet(PRICE_FILE)


def load_prices() -> pd.DataFrame:
    flat = pd.read_parquet(PRICE_FILE)
    flat.columns = pd.MultiIndex.from_tuples([tuple(c.split("|", 1)) for c in flat.columns])
    return flat
