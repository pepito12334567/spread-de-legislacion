"""Descarga y parseo de los breakpoints oficiales de Ken French (percentiles NYSE)."""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from .config import CACHE, FRENCH_BASE, PCTS

FILES = {
    "ME": "ME_Breakpoints_CSV",
    "BM": "BE-ME_Breakpoints_CSV",
    "PRIOR_2_12": "Prior_2-12_Breakpoints_CSV",
}
_ROW = re.compile(r"^\s*(\d{4,6})\s*,")


def _download(stem: str, refresh: bool = False) -> str:
    path = CACHE / f"french_{stem}.zip"
    if refresh or not path.exists():
        r = requests.get(f"{FRENCH_BASE}/{stem}.zip", timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def parse_breakpoints(text: str, var: str) -> pd.DataFrame:
    """Devuelve DataFrame indexado por fecha (YYYYMM o YYYY) con columnas n y p5..p100.

    ME: valores en millones de USD. BM: ratio. PRIOR_2_12: retorno en %.
    """
    rows = []
    for line in text.splitlines():
        if not _ROW.match(line):
            continue
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        key = int(parts[0])
        if var == "BM":  # año, n(BE<=0), n(BE>0), 20 percentiles
            n, vals = int(parts[2]), parts[3:]
        else:            # fecha, n, 20 percentiles
            n, vals = int(parts[1]), parts[2:]
        vals = [float(v) for v in vals][-len(PCTS):]
        if len(vals) != len(PCTS):
            continue
        rows.append([key, n, *vals])
    df = pd.DataFrame(rows, columns=["key", "n", *[f"p{p}" for p in PCTS]]).set_index("key")
    return df


def load_breakpoints(var: str, refresh: bool = False) -> pd.DataFrame:
    return parse_breakpoints(_download(FILES[var], refresh), var)


def get_cuts(df: pd.DataFrame, key: int, cuts: list[int]) -> list[float] | None:
    """Valores de breakpoint para los percentiles pedidos (múltiplos de 5)."""
    if key not in df.index:
        return None
    row = df.loc[key]
    return [float(row[f"p{c}"]) for c in cuts]


def load_portfolio_returns(stem: str, refresh: bool = False) -> str:
    """Texto crudo de un archivo de retornos de portfolios (para validación futura)."""
    return _download(stem, refresh)
