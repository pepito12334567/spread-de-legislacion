"""Replicación de retornos mensuales VW de los portfolios de French con datos gratuitos (solo sobrevivientes)."""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from . import engine, french, variables


def parse_french_returns(text: str, section: str = "Average Value Weighted Returns -- Monthly") -> pd.DataFrame:
    """Lee una sección mensual de un archivo de retornos de French. Índice = periodo AAAA-MM."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if section.lower() in l.lower())
    header = [h.strip() for h in lines[start + 1].split(",")][1:]
    rows = []
    for l in lines[start + 2:]:
        if not re.match(r"^\s*\d{6}\s*,", l):
            break
        p = [x.strip() for x in l.split(",")]
        rows.append([pd.Period(p[0], "M")] + [float(x) for x in p[1:]])
    df = pd.DataFrame(rows, columns=["period", *header]).set_index("period")
    return df.replace([-99.99, -999], np.nan)


def month_end_prices(adj: pd.DataFrame) -> pd.DataFrame:
    return adj.ffill(limit=5).resample("ME").last()


def _vw_return(w0: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """Retorno VW mensual con pesos que derivan con los retornos (equivale a pesos = ME de inicio de mes)."""
    r = rets.fillna(0.0)
    growth = (1 + r).cumprod().shift(1).fillna(1.0)
    w = growth.mul(w0, axis=1)
    return (w * r).sum(axis=1) / w.sum(axis=1)


def replicate_annual(data, spec: engine.Spec, june_year: int, source: str = "french", months: int = 12) -> pd.DataFrame:
    """Formación a fin de junio de `june_year`; devuelve retornos mensuales (%) por celda para los `months` siguientes."""
    universe, px, shares, be = data
    F = pd.Timestamp(year=june_year, month=6, day=30)
    t = variables.build_table(universe, px, shares, be, F)
    cuts = [engine.resolve_cuts(t, s, F, source)[0] for s in spec.sorts]
    cl = engine.classify(t, spec, cuts).dropna(subset=["cell"])
    mp = month_end_prices(px["Adj Close"])
    rets = mp.pct_change().loc[F + pd.offsets.MonthEnd(1): F + pd.offsets.MonthEnd(months)]
    out = {}
    for cell, g in cl.groupby("cell"):
        cols = [c for c in g["ticker"] if c in rets.columns]
        w0 = g.set_index("ticker")["ME"].reindex(cols)
        out[cell] = _vw_return(w0, rets[cols]) * 100
    df = pd.DataFrame(out)
    df.index = df.index.to_period("M")
    return df[[c for c in spec.cells if c in df]]


def replicate_monthly(data, spec: engine.Spec, formation_months: list[pd.Timestamp], source: str = "french") -> pd.DataFrame:
    """Formación a fin de cada mes en `formation_months`; el retorno corresponde al mes siguiente."""
    universe, px, shares, be = data
    mp = month_end_prices(px["Adj Close"])
    rets = mp.pct_change()
    rows = {}
    for F in formation_months:
        F = variables.month_end(F)
        t = variables.build_table(universe, px, shares, be, F)
        cuts = [engine.resolve_cuts(t, s, F, source)[0] for s in spec.sorts]
        cl = engine.classify(t, spec, cuts).dropna(subset=["cell"])
        nxt = F + pd.offsets.MonthEnd(1)
        if nxt not in rets.index:
            continue
        r = rets.loc[nxt]
        res = {}
        for cell, g in cl.groupby("cell"):
            x = pd.DataFrame({"w": g.set_index("ticker")["ME"], "r": r.reindex(g["ticker"]).to_numpy()}).dropna()
            res[cell] = (x["w"] * x["r"]).sum() / x["w"].sum() * 100
        rows[nxt.to_period("M")] = res
    return pd.DataFrame(rows).T


FRENCH_NAMES = {  # nombre de columna de French -> celda propia
    "size_bm_2x3": {"SMALL LoBM": "Small / Low", "ME1 BM2": "Small / Med", "SMALL HiBM": "Small / High",
                    "BIG LoBM": "Big / Low", "ME2 BM2": "Big / Med", "BIG HiBM": "Big / High"},
    "size_mom_2x3": {"SMALL LoPRIOR": "Small / Loser", "ME1 PRIOR2": "Small / Neutral", "SMALL HiPRIOR": "Small / Winner",
                     "BIG LoPRIOR": "Big / Loser", "ME2 PRIOR2": "Big / Neutral", "BIG HiPRIOR": "Big / Winner"},
}


def compare(rep: pd.DataFrame, french_df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rows = []
    for fcol, cell in mapping.items():
        if fcol not in french_df or cell not in rep:
            continue
        a, b = rep[cell].dropna(), french_df[fcol]
        idx = a.index.intersection(b.dropna().index)
        a, b = a.loc[idx], b.loc[idx]
        rows.append({"celda": cell, "meses": len(idx), "corr": a.corr(b),
                     "acum_propio_%": ((1 + a / 100).prod() - 1) * 100, "acum_french_%": ((1 + b / 100).prod() - 1) * 100,
                     "tracking_error_anual_%": (a - b).std() * np.sqrt(12), "dif_media_mensual_pp": (a - b).mean()})
    return pd.DataFrame(rows).set_index("celda")
