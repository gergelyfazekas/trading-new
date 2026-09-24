"""Chart the biggest winners and losers (by ret_net) among rebound_retest
trades, to look for a visual pattern distinguishing them. Reuses
plot_signal() from two_touch_low_charts.py.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_extremes_chart.py [n_each]
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
CHART_DIR = os.path.join(HERE, "data", "two_touch_low_extremes_charts")


def main():
    n_each = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    df = pd.read_csv(TRADES_CSV, parse_dates=["low_date", "entry_date", "exit_date"])
    rr = df[df.branch == "rebound_retest"].sort_values("ret_net")
    sample = pd.concat([rr.head(n_each), rr.tail(n_each)])
    print(f"{n_each} worst + {n_each} best rebound_retest trades:\n")
    print(sample[["stock", "entry_date", "ret_net"]].to_string(index=False))

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
        tag = "worst" if row.ret_net < 0 else "best"
        out_path = os.path.join(CHART_DIR, f"{tag}_{t}_{row.entry_date.date()}.png")
        plot_signal(t, close_cache[t], volume_cache[t], row, out_path)
        print(f"  {tag:5} {t:6} {row.entry_date.date()} ret_net={row.ret_net:+.1%} -> {out_path}")

    print(f"\nsaved {len(sample)} chart(s) to {CHART_DIR}")


if __name__ == "__main__":
    main()
