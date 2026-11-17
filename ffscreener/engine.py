"""Motor genérico: definición de portfolio -> breakpoints -> asignación de acciones a celdas."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from . import french
from .config import PCTS, PORTFOLIOS_YAML

VAR_LABEL = {"ME": "Market cap (M USD)", "BM": "Book-to-market", "PRIOR_2_12": "Prior 12-2 (%)"}


@dataclass
class Sort:
    var: str
    cuts: list[int]
    labels: list[str]


@dataclass
class Spec:
    key: str
    label: str
    rebalance: str
    sorts: list[Sort]

    @property
    def cells(self) -> list[str]:
        return [" / ".join(c) for c in itertools.product(*[s.labels for s in self.sorts])]


def load_specs() -> dict[str, Spec]:
    raw = yaml.safe_load(PORTFOLIOS_YAML.read_text())["portfolios"]
    return {k: Spec(k, v["label"], v["rebalance"], [Sort(**s) for s in v["sorts"]]) for k, v in raw.items()}


def french_key(var: str, formation: pd.Timestamp) -> int:
    """Clave de fila en los archivos de French: AAAA (BM, formación de junio) o AAAAMM (fin de mes)."""
    from .variables import june_formation_year
    return june_formation_year(formation) if var == "BM" else formation.year * 100 + formation.month


_FRENCH = {}


def published_cuts(var: str, cuts: list[int], formation: pd.Timestamp) -> list[float] | None:
    if var not in _FRENCH:
        _FRENCH[var] = french.load_breakpoints(var)
    return french.get_cuts(_FRENCH[var], french_key(var, formation), cuts)


def own_cuts(table: pd.DataFrame, var: str, cuts: list[int]) -> list[float]:
    """Percentiles calculados con la muestra NYSE propia (mismo criterio que French: solo NYSE)."""
    m = table["exchange"] == "NYSE"
    if "spac" in table:  # CRSP/Compustat no incluye SPACs en la muestra de breakpoints
        m &= ~table["spac"].fillna(False)
    s = table.loc[m, var].dropna()
    return [float(np.percentile(s, c)) for c in cuts] if len(s) else [np.nan] * len(cuts)


def resolve_cuts(table: pd.DataFrame, sort: Sort, formation: pd.Timestamp, source: str = "auto") -> tuple[list[float], str]:
    """source: 'french' | 'own' | 'auto' (French si está publicado, si no propio)."""
    if source in ("french", "auto"):
        pub = published_cuts(sort.var, sort.cuts, formation)
        if pub is not None:
            return pub, "French (publicado)"
        if source == "french":
            raise ValueError(f"French no publicó {sort.var} para {formation:%Y-%m}")
    return own_cuts(table, sort.var, sort.cuts), "Propio (NYSE, datos gratuitos)"


def classify(table: pd.DataFrame, spec: Spec, cuts: list[list[float]]) -> pd.DataFrame:
    t = table.copy()
    labels = []
    ok = pd.Series(True, index=t.index)
    for s, c in zip(spec.sorts, cuts):
        v = t[s.var]
        ok &= v.notna()
        idx = np.searchsorted(np.asarray(c, dtype=float), v.fillna(0).to_numpy(), side="left")
        labels.append(pd.Series(np.array(s.labels, dtype=object)[idx], index=t.index))
    t["cell"] = pd.concat(labels, axis=1).agg(" / ".join, axis=1).where(ok)
    return t


def published_vs_own(table: pd.DataFrame, formation: pd.Timestamp) -> pd.DataFrame:
    """Compara todos los percentiles (5..100) publicados por French vs. los calculados con datos gratuitos."""
    rows = []
    for var in ("ME", "BM", "PRIOR_2_12"):
        pub = published_cuts(var, PCTS, formation)
        own = own_cuts(table, var, PCTS)
        for p, a, b in zip(PCTS, pub or [np.nan] * len(PCTS), own):
            rows.append({"var": var, "pct": p, "french": a, "propio": b})
    df = pd.DataFrame(rows)
    df["dif_%"] = (df["propio"] / df["french"] - 1) * 100
    return df
