"""Filtros de implementación (micro caps), ponderación y órdenes sugeridas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_filters(df: pd.DataFrame, min_price: float = 0.0, min_adv_usd: float = 0.0, max_me: float | None = None,
                  min_me: float = 0.0, exchanges: list[str] | None = None, drop_spac: bool = False,
                  drop_deficient: bool = False) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    m &= df["price"].fillna(0) >= min_price
    m &= df["adv_usd"].fillna(0) >= min_adv_usd
    m &= df["ME"].fillna(0) >= min_me
    if max_me:
        m &= df["ME"].fillna(np.inf) <= max_me
    if exchanges:
        m &= df["exchange"].isin(exchanges)
    if drop_spac:
        m &= ~df["spac"]
    if drop_deficient:
        m &= ~df["deficient"]
    return df[m]


def _cap_weights(raw: np.ndarray, caps: np.ndarray, iters: int = 100) -> np.ndarray:
    """Pesos proporcionales a `raw`, sin superar `caps` (individuales), redistribuyendo el exceso. Suma 1 si es factible."""
    w = raw / raw.sum()
    free = np.ones(len(w), bool)
    for _ in range(iters):
        over = free & (w > caps + 1e-12)
        if not over.any():
            break
        w[over] = caps[over]
        free &= ~over
        rest = 1 - w[~free].sum()
        if free.sum() == 0 or rest <= 0:
            break
        w[free] = raw[free] / raw[free].sum() * rest
    return w


def weights(df: pd.DataFrame, mode: str = "vw", cap: float = 0.05, capital: float = 10_000, participation: float = 0.05) -> pd.Series:
    """mode: vw | ew | vw_cap (tope `cap` por posición) | liq (VW limitado por participation × ADV / capital)."""
    if df.empty:
        return pd.Series(dtype=float)
    me = df["ME"].to_numpy(float)
    if mode == "ew":
        w = np.ones(len(df)) / len(df)
    elif mode == "vw_cap":
        w = _cap_weights(me, np.full(len(df), max(cap, 1 / len(df))))
    elif mode == "liq":
        caps = np.minimum(participation * df["adv_usd"].fillna(0).to_numpy(float) / capital, 1.0)
        caps = np.maximum(caps, 1e-9)
        w = _cap_weights(me, caps)
    else:
        w = me / me.sum()
    return pd.Series(w, index=df.index, name="weight")


def orders(df: pd.DataFrame, w: pd.Series, capital: float, top_n: int | None = None) -> pd.DataFrame:
    d = df.assign(weight=w).sort_values("weight", ascending=False)
    if top_n:
        d = d.head(top_n)
        d["weight"] = d["weight"] / d["weight"].sum()
    d["target_usd"] = d["weight"] * capital
    d["shares_to_buy"] = np.floor(d["target_usd"] / d["price"]).astype("Int64")
    d["invested_usd"] = d["shares_to_buy"] * d["price"]
    d["pct_of_adv"] = d["invested_usd"] / d["adv_usd"].replace(0, np.nan) * 100
    return d
