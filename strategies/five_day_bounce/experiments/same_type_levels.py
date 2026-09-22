"""Experiment: do levels built only from same-kind touch pairs (trough+trough,
peak+peak) trade better than the frozen rule's kind-agnostic pairing?

Same engine, combo, entry/exit rule, cost and universes as
tech_level_naive_strategy.py / tech_level_oos_strategy.py -- the ONLY change
is build_levels_causal(same_type=True). Reports the frozen-rule slice (level
age < MAX_AGE_DAYS, hold 5) plus all-ages, with a matched-count random-entry
placebo per arm. Read-only: writes nothing to data/.

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/experiments/same_type_levels.py
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import config
from stock_class import StockList
from tech_levels import find_touches, build_levels_causal, mark_broken
from tech_level_naive_strategy import (load_fixed_combo, simulate, equal_weight_curve,
                                       attach_benchmark, START_DATE)

MAX_AGE = 5


def levels_for(close, combo, same_type):
    combo = dict(combo)
    w = combo.pop("tech_width")
    for k in ("consider_volume", "volume_height", "volume_prominence"):
        combo.pop(k, None)
    touches = find_touches(close, with_kind=True, **combo)
    levels = build_levels_causal(touches, w, same_type=same_type)
    mark_broken(levels, close)
    return levels


def load(universe):
    if universe == "ticker_list":
        tickers = config.ticker_list
        sl = StockList(tickers)
        sl.pull_data(start=START_DATE, end=datetime.date.today())
    else:
        tickers = [t for t in config.oos_ticker_list if t != "NVDA"]
        sl = StockList(tickers)
        sl.load_data(config.oos)
    out = {}
    for t in tickers:
        try:
            c = sl[t].data["close"].dropna()
            c.index = pd.to_datetime(c.index)
            if not c.empty:
                out[t] = c
        except Exception:
            pass
    return out


def stats(df, label):
    n = len(df)
    if n < 3:
        return f"{label:28} n={n}"
    ex = df.excess_ret
    return (f"{label:28} n={n:5d}  hit={(df.ret > 0).mean():5.1%}  net={df.ret_net.mean():+.3%}  "
            f"excess={ex.mean():+.3%}  t={ex.mean() / ex.std() * np.sqrt(n):+.2f}")


def placebo_excess(series, ew, young, seed=0):
    rng = np.random.default_rng(seed)
    counts = young.groupby("stock").size()
    rows = []
    for t, c in series.items():
        k = int(counts.get(t, 0))
        n = len(c)
        if k == 0 or n < 10:
            continue
        starts = rng.choice(np.arange(0, n - 6), size=min(k, n - 6), replace=False)
        for i in starts:
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + 5],
                         "ret_net": c.iloc[i + 5] / c.iloc[i] - 1 - 10 / 1e4})
    p = pd.DataFrame(rows)
    p = attach_benchmark(p, ew)
    return p.excess_ret.mean()


def main():
    combo = load_fixed_combo()
    for universe in ("ticker_list", "oos"):
        series = load(universe)
        ew = equal_weight_curve(series)
        print(f"\n===== {universe}: {len(series)} tickers =====")
        for name, same in (("baseline (any pair)", False), ("same-type pairs only", True)):
            lv = {t: levels_for(c, combo, same) for t, c in series.items()}
            n_lv = sum(len(v) for v in lv.values())
            trades = []
            for t, c in series.items():
                trades += simulate(t, c, lv[t], hold_days=5, near_pct=0.01)
            df = attach_benchmark(pd.DataFrame(trades), ew)
            young = df[df.support_age_days < MAX_AGE]
            print(f"\n-- {name}: {n_lv} levels")
            print(stats(df, "all ages"))
            print(stats(young, f"age < {MAX_AGE}d (frozen rule)"))
            print(stats(df[df.support_age_days >= MAX_AGE], f"age >= {MAX_AGE}d"))
            if len(young) >= 3:
                pe = placebo_excess(series, ew, young)
                print(f"{'placebo excess (matched n)':28} {pe:+.3%}   lift = {young.excess_ret.mean() - pe:+.3%}")


if __name__ == "__main__":
    main()
