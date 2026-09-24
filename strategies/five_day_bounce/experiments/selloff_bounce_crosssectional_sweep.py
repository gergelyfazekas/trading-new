"""Grid sweep for selloff_bounce_crosssectional_daybyday.py: n_return x
bottom_pct x hold, 20-ticker seed=21 prototype, data pulled once.

Run: ./venv/bin/python strategies/five_day_bounce/experiments/selloff_bounce_crosssectional_sweep.py
"""
import datetime
import os
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from selloff_bounce_crosssectional_daybyday import compute_rank_matrix, run_ticker_args, COST, WARMUP

SEED, K = 21, 20


def main():
    tickers = random.Random(SEED).sample(LIVE_TICKERS, K)
    series, failed = pull_all(tickers)
    today = datetime.date.today()

    close = {}
    for t in tickers:
        if t not in series:
            continue
        c, _ = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]

    ew = equal_weight_curve(close)
    close_df = pd.DataFrame(close).dropna(how="any")

    rng = np.random.default_rng(0)

    grid = []
    for n_return in [5, 10, 20]:
        for bottom_pct in [0.05, 0.10, 0.20]:
            for hold in [5, 10]:
                grid.append(dict(n_return=n_return, bottom_pct=bottom_pct, hold=hold))

    print(f"{'n_ret':>6} {'bot%':>5} {'hold':>5} {'n':>6} {'hit':>7} {'excess':>9} {'t':>7}  {'placebo':>9} {'lift':>9}")
    for p in grid:
        rank_df = compute_rank_matrix(close_df, p["n_return"])
        trades = []
        for t in close_df.columns:
            trades.extend(run_ticker_args(t, close_df, rank_df, p))
        df = pd.DataFrame(trades)
        if df.empty:
            print(f"{p['n_return']:>6} {p['bottom_pct']:>5.2f} {p['hold']:>5}   no trades")
            continue
        df = attach_benchmark(df, ew)

        rows = []
        for t, cnt in df.groupby("stock").size().items():
            c = close[t]
            idxs = rng.choice(np.arange(WARMUP, len(c) - p["hold"]), size=cnt, replace=False)
            for i in idxs:
                rows.append({"entry_date": c.index[i], "exit_date": c.index[i + p["hold"]],
                             "ret_net": c.iloc[i + p["hold"]] / c.iloc[i] - 1 - COST})
        pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

        ex = df.excess_ret
        t_stat = ex.mean() / ex.std() * np.sqrt(len(df))
        print(f"{p['n_return']:>6} {p['bottom_pct']:>5.2f} {p['hold']:>5} {len(df):>6} {(df.ret>0).mean():>7.1%} "
              f"{ex.mean():>+9.3%} {t_stat:>+7.2f}  {pe:>+9.3%} {ex.mean()-pe:>+9.3%}")


if __name__ == "__main__":
    main()
