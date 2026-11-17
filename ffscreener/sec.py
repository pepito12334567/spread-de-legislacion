"""Fundamentales gratuitos desde SEC EDGAR XBRL 'frames' (un valor por empresa y período).

Book equity (definición French): patrimonio + impuestos diferidos (opcional) - acciones preferentes,
del último ejercicio fiscal que cierra en el año calendario t-1.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from .config import CACHE, SEC_HEADERS

BASE = "https://data.sec.gov/api/xbrl/frames"
_last = [0.0]


def fetch_frame(tag: str, unit: str, period: str, taxonomy: str = "us-gaap", refresh: bool = False) -> pd.DataFrame:
    """Un frame XBRL: DataFrame[cik, end, val]. Cacheado en disco."""
    cache = CACHE / "sec" / f"{taxonomy}_{tag}_{unit}_{period}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
    else:
        wait = 0.12 - (time.time() - _last[0])  # SEC permite 10 req/s
        if wait > 0:
            time.sleep(wait)
        r = requests.get(f"{BASE}/{taxonomy}/{tag}/{unit}/{period}.json", headers=SEC_HEADERS, timeout=60)
        _last[0] = time.time()
        if r.status_code == 404:
            data = []
        else:
            r.raise_for_status()
            data = r.json()["data"]
        cache.write_text(json.dumps(data))
    if not data:
        return pd.DataFrame(columns=["cik", "end", "val"])
    df = pd.DataFrame(data)[["cik", "end", "val"]]
    df["end"] = pd.to_datetime(df["end"])
    return df


def _quarters(year: int) -> list[str]:
    return [f"CY{year}Q{q}I" for q in range(1, 5)]


def _instants(tag: str, year: int) -> pd.DataFrame:
    return pd.concat([fetch_frame(tag, "USD", p) for p in _quarters(year)]).drop_duplicates(["cik", "end"])


ANNUAL_TAGS = ["NetIncomeLoss", "ProfitLoss", "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "OperatingIncomeLoss"]


def fiscal_year_ends(year: int) -> pd.DataFrame:
    """Fechas de cierre de ejercicio (períodos anuales) que caen en el año calendario `year`."""
    frames = [fetch_frame(t, "USD", f"CY{y}") for t in ANNUAL_TAGS for y in (year - 1, year)]
    a = pd.concat(frames)[["cik", "end"]].drop_duplicates()
    return a[a["end"].dt.year == year]


def book_equity(year: int, include_deferred_tax: bool = False) -> pd.DataFrame:
    """Book equity del ejercicio fiscal que cierra en `year` (se usa para la formación de junio de year+1).

    Devuelve cik, fye (fecha), be (USD), se, pref, dtl, source.
    """
    se = _instants("StockholdersEquity", year)
    se2 = _instants("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", year)
    se = pd.concat([se, se2]).drop_duplicates(["cik", "end"], keep="first").rename(columns={"val": "se"})
    pref = _instants("PreferredStockValue", year).rename(columns={"val": "pref"})
    dtl = _instants("DeferredIncomeTaxLiabilitiesNet", year).rename(columns={"val": "dtl"})

    fye = fiscal_year_ends(year)
    m = se.merge(fye, on=["cik", "end"], how="inner")
    m["source"] = "fye"
    # Empresas sin período anual identificable: usar balance de diciembre como aproximación.
    rest = se[~se["cik"].isin(m["cik"]) & (se["end"].dt.month == 12)].copy()
    rest["source"] = "dec_fallback"
    m = pd.concat([m, rest])
    m = m.sort_values("end").drop_duplicates("cik", keep="last")  # último cierre del año

    m = m.merge(pref, on=["cik", "end"], how="left").merge(dtl, on=["cik", "end"], how="left")
    m["pref"] = m["pref"].fillna(0.0)
    m["dtl"] = m["dtl"].fillna(0.0)
    m["be"] = m["se"] - m["pref"] + (m["dtl"] if include_deferred_tax else 0.0)
    return m.rename(columns={"end": "fye"})[["cik", "fye", "be", "se", "pref", "dtl", "source"]].reset_index(drop=True)


def shares_outstanding(latest_year: int, latest_q: int) -> pd.DataFrame:
    """Acciones en circulación (portada de 10-K/10-Q, dei) más recientes por CIK."""
    periods, y, q = [], latest_year, latest_q
    for _ in range(6):
        periods.append(f"CY{y}Q{q}I")
        q -= 1
        if q == 0:
            y, q = y - 1, 4
    df = pd.concat([fetch_frame("EntityCommonStockSharesOutstanding", "shares", p, taxonomy="dei") for p in periods])
    df = df.sort_values("end").drop_duplicates("cik", keep="last")
    return df.rename(columns={"end": "shares_date", "val": "shares"})[["cik", "shares", "shares_date"]]
