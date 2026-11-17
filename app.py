"""Screener interactivo de portfolios Fama-French (datos gratuitos).  Ejecutar: streamlit run app.py"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from ffscreener import engine, portfolio, variables
from ffscreener.config import CACHE, PCTS

st.set_page_config(page_title="FF Screener", layout="wide")
st.title("Screener de portfolios Fama-French")

if not (CACHE / "prices.parquet").exists():
    st.error("Faltan datos. Ejecutá primero:  `python -m ffscreener.update`")
    st.stop()


@st.cache_data(show_spinner="Cargando datos…")
def get_data():
    return variables.load_data()


@st.cache_data(show_spinner="Calculando variables…")
def get_table(formation: pd.Timestamp):
    universe, px, shares, be = get_data()
    return variables.build_table(universe, px, shares, be, formation)


universe, px, shares, be = get_data()
last_px = px["Close"].dropna(how="all").index.max()
specs = engine.load_specs()

# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("1. Portfolio")
choice = sb.selectbox("Definición", list(specs) + ["custom"], format_func=lambda k: specs[k].label if k in specs else "Personalizado…")

if choice == "custom":
    nsorts = sb.radio("Número de sorts", [1, 2], horizontal=True)
    rebalance = sb.radio("Rebalanceo", ["annual", "monthly"], horizontal=True, format_func=lambda x: {"annual": "Anual (junio)", "monthly": "Mensual"}[x])
    sorts = []
    for i in range(nsorts):
        v = sb.selectbox(f"Variable {i + 1}", list(engine.VAR_LABEL), index=min(i, 2), format_func=engine.VAR_LABEL.get, key=f"v{i}")
        c = sb.text_input(f"Percentiles NYSE (múltiplos de 5) {i + 1}", "50" if v == "ME" else "30,70", key=f"c{i}")
        cuts = sorted({int(x) for x in c.split(",") if x.strip().isdigit() and int(x) % 5 == 0 and 0 < int(x) < 100})
        sorts.append(engine.Sort(v, cuts, [f"Q{j + 1}" for j in range(len(cuts) + 1)]))
    spec = engine.Spec("custom", "Personalizado", rebalance, sorts)
else:
    spec = specs[choice]

sb.header("2. Fecha de formación")
if spec.rebalance == "annual":
    years = sorted(y + 1 for y in be if dt.date(y + 1, 6, 30) <= last_px.date())
    t = sb.selectbox("Formación de junio de", years[::-1])
    formation = pd.Timestamp(year=t, month=6, day=30)
    sb.caption(f"BE: ejercicio fiscal {t - 1} · ME B/M: dic-{t - 1} · ME size: 30-jun-{t}. Se mantiene hasta jun-{t + 1}.")
else:
    ends = pd.date_range(end=last_px, periods=14, freq="ME")
    ends = [e for e in ends if e <= last_px.normalize()]
    formation = sb.selectbox("Fin de mes de formación (se mantiene el mes siguiente)", ends[::-1], format_func=lambda d: f"{d:%Y-%m-%d}")
    formation = pd.Timestamp(formation)

source = sb.radio("Breakpoints", ["auto", "french", "own"], format_func={"auto": "Auto (French si existe)", "french": "Solo French publicados", "own": "Propios (NYSE)"}.get)

sb.header("3. Filtros de implementación")
min_price = sb.number_input("Precio mínimo (USD)", 0.0, 1000.0, 1.0, 0.5)
min_adv = sb.number_input("Volumen diario mínimo (USD, prom. 20d)", 0, 100_000_000, 50_000, 10_000)
max_me = sb.number_input("Market cap máximo (M USD, 0 = sin tope)", 0.0, 1e7, 0.0, 50.0)
exch = sb.multiselect("Exchanges", ["NYSE", "AMEX", "NASDAQ"], ["NYSE", "AMEX", "NASDAQ"])
drop_spac = sb.checkbox("Excluir SPACs", True)
drop_def = sb.checkbox("Excluir con deficiencia de listado (Nasdaq)", False)

sb.header("4. Ponderación y capital")
mode = sb.selectbox("Pesos", ["vw", "vw_cap", "liq", "ew"], format_func={
    "vw": "Value-weighted (como French)", "vw_cap": "VW con tope por posición",
    "liq": "VW limitado por liquidez", "ew": "Igual ponderado"}.get)
capital = sb.number_input("Capital (USD)", 100, 100_000_000, 10_000, 1_000)
cap = sb.slider("Tope por posición (%)", 1, 30, 5) / 100 if mode == "vw_cap" else 0.05
part = sb.slider("Máx. % del volumen diario por posición", 1, 25, 5) / 100 if mode == "liq" else 0.05
top_n = sb.number_input("Limitar a las N de mayor peso (0 = todas)", 0, 1000, 0, 5)

# ---------------- Cálculo ----------------
table = get_table(formation)
cuts, srcs = [], []
for s in spec.sorts:
    c, src = engine.resolve_cuts(table, s, formation, source) if source != "own" else (engine.own_cuts(table, s.var, s.cuts), "Propio (NYSE, datos gratuitos)")
    cuts.append(c)
    srcs.append(src)
cl = engine.classify(table, spec, cuts)

tabs = st.tabs(["Breakpoints", "Celdas", "Portfolio", "Universo", "Validación"])

with tabs[0]:
    st.subheader(f"Breakpoints usados — formación {formation:%d-%b-%Y}")
    rows = []
    for s, c, src in zip(spec.sorts, cuts, srcs):
        for p, v in zip(s.cuts, c):
            rows.append({"variable": engine.VAR_LABEL[s.var], "percentil NYSE": p, "valor": v, "fuente": src})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Un valor igual al breakpoint cae en el grupo inferior. Los de French se publican con rezago de ~1-2 meses; cuando no existen se calculan con acciones NYSE propias.")

with tabs[1]:
    st.subheader("Cantidad de acciones y % del market cap por celda")
    ok = cl.dropna(subset=["cell"])
    g = ok.groupby("cell").agg(acciones=("ticker", "size"), mcap_musd=("ME", "sum"))
    g["% mcap"] = g["mcap_musd"] / g["mcap_musd"].sum() * 100
    g = g.reindex(spec.cells)
    if len(spec.sorts) == 2:
        l1, l2 = spec.sorts[0].labels, spec.sorts[1].labels
        for col, fmt in (("acciones", "{:,.0f}"), ("% mcap", "{:.1f}%")):
            st.markdown(f"**{col}**")
            m = pd.DataFrame([[g.loc[f"{a} / {b}", col] for b in l2] for a in l1], index=l1, columns=l2)
            st.dataframe(m.style.format(fmt, na_rep="–"), use_container_width=True)
    else:
        st.dataframe(g, use_container_width=True)
    st.caption(f"{len(ok):,} de {len(cl):,} acciones del universo con todas las variables disponibles.")

with tabs[2]:
    cell = st.selectbox("Celda a replicar", spec.cells, index=min(len(spec.cells) - 1, 2) if len(spec.cells) > 2 else 0)
    members = cl[cl["cell"] == cell]
    sel = portfolio.apply_filters(members, min_price, min_adv, max_me or None, 0.0, exch, drop_spac, drop_def)
    c1, c2, c3 = st.columns(3)
    c1.metric("Miembros según French", f"{len(members):,}")
    c2.metric("Tras tus filtros", f"{len(sel):,}")
    c3.metric("% del market cap que conservás", f"{sel['ME'].sum() / members['ME'].sum() * 100:.1f}%" if len(members) else "–")
    if len(sel):
        w = portfolio.weights(sel, mode, cap, capital, part)
        o = portfolio.orders(sel, w, capital, int(top_n) or None)
        o["peso_%"] = o["weight"] * 100
        show = o[["ticker", "name", "exchange", "price", "ME", "BM", "PRIOR_2_12", "adv_usd", "peso_%", "target_usd", "shares_to_buy", "invested_usd", "pct_of_adv"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Posiciones", len(o))
        c2.metric("Invertido", f"${o['invested_usd'].sum():,.0f}")
        c3.metric("Máx. % del volumen diario", f"{o['pct_of_adv'].max():.1f}%")
        st.dataframe(show.style.format({"price": "{:.2f}", "ME": "{:,.0f}", "BM": "{:.2f}", "PRIOR_2_12": "{:.1f}", "adv_usd": "{:,.0f}",
                                        "peso_%": "{:.2f}", "target_usd": "{:,.0f}", "invested_usd": "{:,.0f}", "pct_of_adv": "{:.1f}"}, na_rep="–"),
                     hide_index=True, use_container_width=True, height=520)
        st.download_button("Descargar órdenes (CSV)", show.to_csv(index=False).encode(), f"ordenes_{spec.key}_{formation:%Y%m%d}.csv", "text/csv")
        st.warning("Usá siempre órdenes limitadas en micro caps y verificá a mano los fundamentales de las mayores posiciones.")
    else:
        st.info("Ninguna acción cumple tus filtros en esta celda.")

with tabs[3]:
    st.subheader("Universo completo con variables y celda asignada")
    q = st.text_input("Buscar ticker o nombre")
    d = cl[["ticker", "name", "exchange", "cell", "price", "ME", "BM", "PRIOR_2_12", "adv_usd", "be_source", "multi_class", "spac"]]
    if q:
        d = d[d["ticker"].str.contains(q, case=False) | d["name"].str.contains(q, case=False)]
    st.dataframe(d, hide_index=True, use_container_width=True, height=600)
    st.caption("multi_class: la firma tiene varias clases cotizando; el ME (acciones de portada SEC) puede estar subestimado.")

with tabs[4]:
    st.subheader("Breakpoints publicados por French vs. calculados con datos gratuitos (NYSE)")
    cmp = engine.published_vs_own(table, formation)
    piv = cmp.pivot(index="pct", columns="var", values="dif_%")
    st.markdown("**Diferencia % (propio vs. French)** — cuanto más cerca de 0, más fiel es tu universo")
    st.dataframe(piv.style.format("{:+.1f}%", na_rep="–"), use_container_width=True)
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.caption("Si no hay fila de French para esta fecha (publicación con rezago) la comparación queda vacía.")
