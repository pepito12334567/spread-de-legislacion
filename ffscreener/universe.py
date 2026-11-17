"""Universo: acciones comunes listadas en NYSE / AMEX / NASDAQ (aprox. CRSP shrcd 10, 11).

Fuente: archivos públicos de NASDAQ Trader (símbolo, nombre, exchange, flag ETF) + mapa ticker->CIK de la SEC.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

from .config import CACHE, SEC_HEADERS

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

# Nombre que indica acción común / ordinaria
_INCLUDE = re.compile(r"common (stock|shares)|ordinary shares|capital stock|class [a-z] (stock|shares)", re.I)
# Nombre que indica otro tipo de valor
_EXCLUDE = re.compile(
    r"warrant|right|unit|preferred|depositary|\bADS\b|\bADR\b|note|bond|debenture|trust|"
    r"%|subordinated|series [a-z]\b|beneficial|partnership|fund|etf|etn|index|contingent",
    re.I,
)
_SPAC = re.compile(r"acquisition (corp|co|holdings)|blank check|capital corp\b.*acquisition", re.I)


def is_common_stock(name: str) -> bool:
    return bool(_INCLUDE.search(name)) and not _EXCLUDE.search(name)


def _read(url: str) -> pd.DataFrame:
    txt = requests.get(url, timeout=60).text
    lines = [l for l in txt.splitlines() if not l.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)


def build_universe(refresh: bool = False) -> pd.DataFrame:
    path = CACHE / "universe.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    nq = _read(NASDAQ_URL).rename(columns={"Symbol": "symbol", "Security Name": "name"})
    nq = nq[(nq["Test Issue"] == "N") & (nq["ETF"] == "N")]
    nq = nq.assign(exchange="NASDAQ", deficient=nq["Financial Status"].isin(["D", "E", "Q", "G", "H", "J", "K"]))

    ot = _read(OTHER_URL).rename(columns={"ACT Symbol": "symbol", "Security Name": "name", "Exchange": "ex"})
    ot = ot[(ot["Test Issue"] == "N") & (ot["ETF"] == "N") & ot["ex"].isin(["N", "A"])]
    ot = ot.assign(exchange=ot["ex"].map({"N": "NYSE", "A": "AMEX"}), deficient=False)

    u = pd.concat([nq[["symbol", "name", "exchange", "deficient"]], ot[["symbol", "name", "exchange", "deficient"]]])
    u = u[u["name"].map(is_common_stock)].copy()
    u["ticker"] = u["symbol"].str.replace(".", "-", regex=False)  # formato Yahoo/SEC (BRK-B)
    u["spac"] = u["name"].str.contains(_SPAC)

    sec = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=60).json()
    sec = pd.DataFrame(sec["data"], columns=sec["fields"])[["cik", "ticker"]]
    sec["ticker"] = sec["ticker"].str.upper()
    u = u.merge(sec, on="ticker", how="left")
    u = u.dropna(subset=["cik"]).drop_duplicates("ticker")
    u["cik"] = u["cik"].astype(int)
    # varias clases cotizando bajo el mismo CIK -> el ME de la firma requiere sumar clases
    u["multi_class"] = u.groupby("cik")["ticker"].transform("size") > 1
    u = u.reset_index(drop=True)
    u.to_parquet(path)
    return u
