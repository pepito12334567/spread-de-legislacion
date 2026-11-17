"""Actualiza todos los datos gratuitos: python -m ffscreener.update [--skip-prices] [--start 2024-12-01]"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from . import french, prices, sec, universe
from .config import CACHE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-12-01", help="inicio de precios diarios")
    ap.add_argument("--be-years", nargs="+", type=int, default=None, help="años fiscales de book equity")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--refresh-universe", action="store_true")
    a = ap.parse_args()
    today = dt.date.today()
    be_years = a.be_years or [today.year - 2, today.year - 1]

    print("1/4 breakpoints de French")
    for v in french.FILES:
        french.load_breakpoints(v, refresh=True)

    print("2/4 universo")
    u = universe.build_universe(refresh=a.refresh_universe)
    print(f"  {len(u)} acciones comunes ({u.exchange.value_counts().to_dict()})")

    print("3/4 SEC: book equity y acciones")
    for y in be_years:
        be = sec.book_equity(y)
        be.to_parquet(CACHE / f"be_{y}.parquet")
        print(f"  BE FY{y}: {len(be)} empresas")
    q = (today.month - 1) // 3 + 1
    sh = sec.shares_outstanding(today.year, q)
    sh.to_parquet(CACHE / "shares.parquet")
    print(f"  acciones: {len(sh)} empresas")

    if not a.skip_prices:
        print("4/4 precios (yfinance, tarda ~10-15 min)")
        px = prices.download_prices(u["ticker"].tolist(), start=a.start)
        prices.save_prices(px)
        print(f"  precios: {px.shape}")
    print("listo")


if __name__ == "__main__":
    main()
