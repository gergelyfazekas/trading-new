"""Chart a ticker's current in-progress two-touch-low candidate (if any) --
same three candidates two_touch_low_live.py's "STILL DEVELOPING" section
diagnoses (low today, low 1 session ago, and today's fully-resolved
"today's candidate" 2 sessions back), picks whichever is most advanced,
and plots it with plot_pending() from two_touch_low_charts.py.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_pending_chart.py TICKER
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
from two_touch_low_daybyday import PARAMS, STAGE_RANK, diagnose
from two_touch_low_charts import plot_pending, CHART_DIR

RETEST_GAP = 2


def main():
    if len(sys.argv) < 2:
        print("usage: two_touch_low_pending_chart.py TICKER")
        return
    ticker = sys.argv[1].upper()

    series, failed = pull_all([ticker])
    if ticker not in series:
        print(f"no data for {ticker}")
        return
    close, volume = series[ticker]
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    if volume is not None:
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)
        volume = volume.reindex(close.index)
    else:
        volume = pd.Series(np.nan, index=close.index)

    n = len(close)
    candidates = [
        (n - 1, "low today"),
        (n - 2, "low 1 session ago"),
        (n - 1 - RETEST_GAP, "today's candidate (2 sessions ago)"),
    ]
    diags = [(L, tag, diagnose(close, volume, L, PARAMS, retest_gap=RETEST_GAP)) for L, tag in candidates]
    for L, tag, d in diags:
        print(f"  L={close.index[L].date()} ({tag}): {d}")

    L, tag, diag = min(diags, key=lambda x: STAGE_RANK.get(x[2]["stage"], 99))
    if diag["stage"] in ("not_a_low", "insufficient_history"):
        print(f"\n{ticker}: nothing active right now (best stage: {diag['stage']})")
        return

    os.makedirs(CHART_DIR, exist_ok=True)
    out_path = os.path.join(CHART_DIR, f"{ticker}_{close.index[-1].date()}_pending.png")
    plot_pending(ticker, close, volume, L, diag, out_path)
    print(f"\n{ticker}: charted {tag} (L={close.index[L].date()}, stage={diag['stage']}) -> {out_path}")


if __name__ == "__main__":
    main()
