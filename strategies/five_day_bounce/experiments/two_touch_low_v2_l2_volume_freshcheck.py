"""Out-of-sample check of two_touch_low_v2_l2_volume_cross.py's findings
(volume_ratio[L2] > 1.0 alone, t=+2.09 on the seed=21/k=20 sample; the
bottom_third & vol>1.0 combo, t=+1.74) on the 81 LIVE_TICKERS never
touched by any of today's tuning -- same "fresh-81" discipline as the
v1 rule's validation in mean_reversion_notes.md.

Reports, on the fresh set: volume_ratio[L2] > 1.0 alone, bottom_third
alone, and the bottom_third & vol>1.0 combo -- same three cuts run on the
in-sample 20, side by side for direct comparison.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_l2_volume_freshcheck.py
"""
import datetime
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from two_touch_low_v2_daybyday import run_ticker, volume_ratio
from two_touch_low_v2_position_terciles import UNGATED_PARAMS, close_position, tercile

VOL_L2_THRESHOLD = 1.0
IN_SAMPLE_20 = ["KO", "GILD", "EMR", "UPS", "MRK", "CRM", "XOM", "ORCL", "TXN", "COST",
                "QCOM", "MU", "HON", "APD", "MMM", "ABT", "TGT", "BLK", "CVS", "AXP"]


def row(df, label):
    if df.empty or len(df) < 3:
        return dict(label=label, n=len(df), hit=np.nan, net=np.nan, excess=np.nan, t=np.nan)
    ex = df.excess_ret
    return dict(label=label, n=len(df), hit=(df.ret > 0).mean(), net=df.ret_net.mean(),
                excess=ex.mean(), t=ex.mean() / ex.std() * np.sqrt(len(df)))


def print_row(r):
    print(f"  {r['label']:32} n={r['n']:4d}  hit={r['hit']:.1%}  net={r['net']:+.3%}  "
          f"excess={r['excess']:+.3%}  t={r['t']:+.2f}")


def main():
    tickers = [t for t in LIVE_TICKERS if t not in IN_SAMPLE_20]
    print(f"fresh {len(tickers)} tickers (LIVE_TICKERS minus the seed=21/k=20 in-sample set): "
          f"{', '.join(tickers)}\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"], include_hl=True)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, high, low, volume = {}, {}, {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, h, l, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        h = h.copy() if h is not None else pd.Series(np.nan, index=c.index)
        h.index = pd.to_datetime(h.index)
        high[t] = h.reindex(close[t].index)
        l = l.copy() if l is not None else pd.Series(np.nan, index=c.index)
        l.index = pd.to_datetime(l.index)
        low[t] = l.reindex(close[t].index)
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)
    spy_close_raw = series["SPY"][0].copy()
    spy_close_raw.index = pd.to_datetime(spy_close_raw.index)
    spy_close = {t: spy_close_raw.reindex(close[t].index, method="ffill") for t in close}

    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], UNGATED_PARAMS, 0.01) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([r for o in out for r in o])
    df = attach_benchmark(df, ew)
    print(f"ungated population on fresh 81: n={len(df)}\n")

    terc_L2, vol_above = [], []
    for _, tr in df.iterrows():
        t = tr.stock
        L2 = close[t].index.get_loc(pd.Timestamp(tr.entry_date))
        terc_L2.append(tercile(close_position(high[t], low[t], close[t], L2)))
        vr = volume_ratio(volume[t], L2)
        vol_above.append((not np.isnan(vr)) and vr > VOL_L2_THRESHOLD)
    df = df.assign(terc_L2=terc_L2, vol_above=vol_above)

    print("=== fresh-81 results (compare to seed=21/k=20 in brackets) ===\n")
    print_row(row(df[df.vol_above], f"vol_L2>{VOL_L2_THRESHOLD} alone"))
    print("    [in-sample: n=153, hit=56.2%, net=+0.668%, t=+2.09]\n")

    print_row(row(df[~df.vol_above], f"vol_L2<={VOL_L2_THRESHOLD} alone"))
    print("    [in-sample: n=244, hit=50.4%, net=+0.230%, t=-0.54]\n")

    print_row(row(df[df.terc_L2 == "bottom_third"], "bottom_third alone"))
    print("    [in-sample: n=249, hit=54.6%, net=+0.591%, t=+0.90]\n")

    combo = df[(df.terc_L2 == "bottom_third") & (df.vol_above)]
    print_row(row(combo, "bottom_third & vol_L2>1.0"))
    print("    [in-sample: n=90, hit=56.7%, net=+0.746%, t=+1.74]")


if __name__ == "__main__":
    main()
