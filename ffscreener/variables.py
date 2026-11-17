"""Construcción de variables (ME, BM, PRIOR_2_12) a una fecha de formación, con datos gratuitos."""
from __future__ import annotations

import pandas as pd

from .config import CACHE


def last_price(df: pd.DataFrame, date: pd.Timestamp, max_stale_days: int = 10) -> pd.Series:
    """Último precio observado en o antes de `date` (máx. `max_stale_days` ruedas de antigüedad)."""
    sub = df.loc[:date].tail(max_stale_days)
    return sub.ffill().iloc[-1] if len(sub) else pd.Series(dtype=float)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(0)).normalize()


def june_formation_year(formation: pd.Timestamp) -> int:
    """Año t de la última formación anual (fin de junio) vigente a `formation`."""
    return formation.year if formation.month >= 6 else formation.year - 1


def split_factor(px: pd.DataFrame, since: pd.Series) -> pd.Series:
    """Producto de ratios de split posteriores a la fecha de las acciones (portada SEC).

    El precio `Close` está ajustado por TODOS los splits hasta hoy; las acciones de la portada no, si el split
    ocurrió después. Multiplicarlas por este factor las deja en la misma base que el precio histórico
    (sin esto, una reverse split deja el ME inflado: acciones viejas × precio ajustado hacia arriba).
    """
    out = pd.Series(1.0, index=since.index)
    if "Stock Splits" not in px.columns.get_level_values(0):
        return out
    sp = px["Stock Splits"]
    sp = sp.loc[:, sp.fillna(0).ne(0).any()]
    for tk in sp.columns.intersection(since.index):
        d = since.get(tk)
        if pd.isna(d):
            continue
        r = sp[tk].loc[pd.Timestamp(d) + pd.Timedelta(days=1):]
        r = r[r.fillna(0) > 0]
        if len(r):
            out[tk] = float(r.prod())
    return out


def build_table(universe: pd.DataFrame, px: pd.DataFrame, shares: pd.DataFrame, be: dict[int, pd.DataFrame],
                formation: pd.Timestamp) -> pd.DataFrame:
    """Una fila por acción con las variables de ordenamiento a la fecha de formación.

    ME      = acciones (portada SEC más reciente) × precio ajustado por splits al cierre de `formation`  [M USD]
    BM      = BE del ejercicio fiscal t-1 / ME de diciembre t-1  (t = última formación de junio)
    PRIOR   = retorno total acumulado desde fin de (F-12m) hasta fin de (F-1m), en %  (se salta el último mes)
    """
    formation = month_end(formation)
    t = june_formation_year(formation)
    close, adj, vol = px["Close"], px["Adj Close"], px["Volume"]

    u = universe.merge(shares, on="cik", how="left")
    u = u.set_index("ticker")
    u["shares"] = u["shares"] * split_factor(px, u["shares_date"])

    p_f = last_price(close, formation)
    u["price"] = p_f.reindex(u.index)
    u["me_musd"] = u["shares"] * u["price"] / 1e6

    dec = pd.Timestamp(year=t - 1, month=12, day=31)
    p_dec = last_price(close, dec).reindex(u.index)
    me_dec = u["shares"] * p_dec / 1e6
    b = be.get(t - 1)
    if b is not None:
        b = b.drop_duplicates("cik").set_index("cik")
        u["be_musd"] = u["cik"].map(b["be"]) / 1e6
        u["fye"] = u["cik"].map(b["fye"])
        u["be_source"] = u["cik"].map(b["source"])
    else:
        u["be_musd"], u["fye"], u["be_source"] = float("nan"), pd.NaT, None
    u["me_dec_musd"] = me_dec
    u["BM"] = (u["be_musd"] / me_dec).where(u["be_musd"] > 0)

    a_end = last_price(adj, month_end(formation - pd.offsets.MonthEnd(1)))
    a_start = last_price(adj, month_end(formation - pd.offsets.MonthEnd(12)))
    u["PRIOR_2_12"] = ((a_end / a_start - 1) * 100).reindex(u.index)

    dv = (close * vol).loc[:formation].tail(20).mean()
    u["adv_usd"] = dv.reindex(u.index)
    u["ME"] = u["me_musd"]
    return u.reset_index()


def load_data():
    """Carga todo lo cacheado por `python -m ffscreener.update`."""
    from . import prices
    universe = pd.read_parquet(CACHE / "universe.parquet")
    shares = pd.read_parquet(CACHE / "shares.parquet")
    be = {int(p.stem.split("_")[1]): pd.read_parquet(p) for p in CACHE.glob("be_*.parquet")}
    px = prices.load_prices()
    return universe, px, shares, be
