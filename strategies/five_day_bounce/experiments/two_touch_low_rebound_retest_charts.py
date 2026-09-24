"""20 randomly sampled rebound_retest (retest_gap=2, locked params) buy
signals, charted the same way as two_touch_low_charts.py -- reuses its
plot_signal() so the plotting logic has one source of truth -- but with a
plain random sample (no win/loss stratification) since the point here is
"what does this branch typically look like", not a balanced before/after
comparison.

Reads two_touch_low_daybyday_trades.csv (the full 101-ticker run).

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_rebound_retest_charts.py [n] [seed]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_live import pull_all
from two_touch_low_charts import plot_signal

TRADES_CSV = os.path.join(HERE, "two_touch_low_daybyday_trades.csv")
CHART_DIR = os.path.join(HERE, "data", "two_touch_low_rebound_retest_charts")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    df = pd.read_csv(TRADES_CSV, parse_dates=["low_date", "entry_date", "exit_date"])
    rr = df[df.branch == "rebound_retest"]
    sample = rr.sample(min(n, len(rr)), random_state=seed)
    print(f"sampled {len(sample)} of {len(rr)} rebound_retest trades "
          f"({(sample.ret_net > 0).mean():.0%} winners in the sample)\n")

    tickers = sorted(sample.stock.unique())
    series, failed = pull_all(tickers)
    if failed:
        print(f"no data for: {failed}")

    os.makedirs(CHART_DIR, exist_ok=True)
    close_cache, volume_cache = {}, {}
    for t in tickers:
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close_cache[t] = c
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume_cache[t] = v.reindex(c.index)
        else:
            volume_cache[t] = pd.Series(np.nan, index=c.index)

    for _, row in sample.iterrows():
        t = row.stock
        out_path = os.path.join(CHART_DIR, f"{t}_{row.entry_date.date()}_rebound_retest.png")
        plot_signal(t, close_cache[t], volume_cache[t], row, out_path)
        print(f"  {t} {row.entry_date.date()} (ret_net={row.ret_net:+.2%}) -> {out_path}")

    print(f"\nsaved {len(sample)} chart(s) to {CHART_DIR}")


if __name__ == "__main__":
    main()
