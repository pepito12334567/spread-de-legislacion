import numpy as np
import pandas as pd

from ffscreener import engine, french, portfolio, universe

ME_TXT = """header
192512,   488,        1.40,        2.38,        3.37,        4.96,        5.95,        7.40,        8.92,       10.81,       13.01,       15.56,       18.50,       22.88,       29.55,       38.41,       49.93,       65.40,       92.72,      142.48,      238.78,     1319.00
"""
BM_TXT = """  ,<= 0,>0
1926,     3,   428,       0.296,       0.408,       0.506,       0.602,       0.662,       0.705,       0.792,       0.863,       0.947,       1.032,       1.094,       1.266,       1.376,       1.550,       1.779,       2.027,       2.456,       3.381,       5.851,      31.324
"""


def test_parse_me():
    df = french.parse_breakpoints(ME_TXT, "ME")
    assert df.loc[192512, "n"] == 488
    assert df.loc[192512, "p50"] == 15.56  # p50 = 10º valor
    assert df.loc[192512, "p100"] == 1319.00
    assert df.shape[1] == 21


def test_parse_bm_skips_nonpositive_count():
    df = french.parse_breakpoints(BM_TXT, "BM")
    assert df.loc[1926, "n"] == 428
    assert df.loc[1926, "p5"] == 0.296 and df.loc[1926, "p100"] == 31.324


def test_classify_2x3_boundaries():
    spec = engine.load_specs()["size_bm_2x3"]
    t = pd.DataFrame({"ME": [10, 100, 100, 10, np.nan], "BM": [0.1, 0.5, 1.0, 1.0, 1.0]})
    out = engine.classify(t, spec, [[50.0], [0.3, 0.7]])
    assert list(out["cell"].iloc[:4]) == ["Small / Low", "Big / Med", "Big / High", "Small / High"]
    assert pd.isna(out["cell"].iloc[4])
    # valor == breakpoint cae en el grupo inferior
    t2 = pd.DataFrame({"ME": [50.0], "BM": [0.7]})
    assert engine.classify(t2, spec, [[50.0], [0.3, 0.7]])["cell"].iloc[0] == "Small / Med"


def test_all_cells_listed():
    specs = engine.load_specs()
    assert len(specs["size_bm_5x5"].cells) == 25
    assert "Small / High" in specs["size_bm_2x3"].cells
    assert len(specs["size_mom_2x3"].cells) == 6


def test_own_cuts_uses_nyse_only():
    t = pd.DataFrame({"exchange": ["NYSE"] * 3 + ["NASDAQ"] * 3, "ME": [1, 2, 3, 100, 200, 300.0]})
    assert engine.own_cuts(t, "ME", [50]) == [2.0]


def test_weights_modes():
    df = pd.DataFrame({"ME": [900.0, 90.0, 10.0], "adv_usd": [1e6, 1e6, 1e3], "price": [10, 10, 1.0]})
    assert np.isclose(portfolio.weights(df, "vw").sum(), 1)
    assert np.allclose(portfolio.weights(df, "ew"), 1 / 3)
    wc = portfolio.weights(df, "vw_cap", cap=0.5)
    assert wc.max() <= 0.5 + 1e-9 and np.isclose(wc.sum(), 1)
    wl = portfolio.weights(df, "liq", capital=10_000, participation=0.05)
    assert wl.iloc[2] <= 0.05 * 1e3 / 10_000 + 1e-9  # el ilíquido queda limitado a 0.5%


def test_orders_floor_shares():
    df = pd.DataFrame({"ME": [3.0, 1.0], "adv_usd": [1e6, 1e6], "price": [7.0, 3.0]})
    o = portfolio.orders(df, portfolio.weights(df), capital=1000)
    assert (o["invested_usd"] <= o["target_usd"] + 1e-9).all()


def test_common_stock_filter():
    assert universe.is_common_stock("Apple Inc. - Common Stock")
    assert universe.is_common_stock("Alphabet Inc. - Class A Common Stock")
    assert not universe.is_common_stock("Foo Corp - Warrant")
    assert not universe.is_common_stock("Bar Inc. - American Depositary Shares")
    assert not universe.is_common_stock("Baz - 6.5% Series A Cumulative Preferred Stock")
    assert not universe.is_common_stock("Qux Acquisition Corp - Units")


def test_split_factor_reverse_split_after_cover_date():
    from ffscreener import variables
    idx = pd.date_range("2026-01-01", periods=5)
    sp = pd.DataFrame({"A": [0, 0, 0.1, 0, 0], "B": [0, 2.0, 0, 0, 0], "C": 0.0}, index=idx)
    px = pd.concat({"Stock Splits": sp}, axis=1)
    since = pd.Series(pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-02"]), index=["A", "B", "C"])
    f = variables.split_factor(px, since)
    assert np.isclose(f["A"], 0.1)   # reverse split posterior: acciones x0.1 (mismas bases que el precio ajustado)
    assert f["B"] == 1.0             # el split 2:1 fue el mismo día: ya está en las acciones de la portada
    assert f["C"] == 1.0
