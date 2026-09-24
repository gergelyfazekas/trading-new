"""Follow-up on two_touch_low_v2_position_terciles.py: does requiring
volume_ratio[L2] > 1.0 sharpen the bottom-third finding, or is it
independent of / in tension with it? Cross-tabs L2's close-position
tercile against L2's volume_ratio (>1.0 or not) on the same ungated
population (Stage 1 + Stage 2 only, no candle/volume gate applied) so
every cell is a clean subset of the same underlying trades -- nothing
here is re-fit or cherry-picked per cell.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_l2_volume_cross.py [seed] [k]
"""
import datetime
import os
import random
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
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}\n", flush=True)

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
    print(f"ungated population: n={len(df)}\n")

    terc_L2, vr_L2, vol_above = [], [], []
    for _, tr in df.iterrows():
        t = tr.stock
        L2 = close[t].index.get_loc(pd.Timestamp(tr.entry_date))
        pos = close_position(high[t], low[t], close[t], L2)
        terc_L2.append(tercile(pos))
        vr = volume_ratio(volume[t], L2)
        vr_L2.append(vr)
        vol_above.append((not np.isnan(vr)) and vr > VOL_L2_THRESHOLD)
    df = df.assign(terc_L2=terc_L2, vr_L2=vr_L2, vol_above=vol_above)

    print(f"=== L2 close-position tercile x volume_ratio[L2] > {VOL_L2_THRESHOLD} ===\n")
    order = ["bottom_third", "middle_third", "top_third"]
    for g in order:
        for va, tag in ((True, f"vol>{VOL_L2_THRESHOLD}"), (False, f"vol<={VOL_L2_THRESHOLD}")):
            sub = df[(df.terc_L2 == g) & (df.vol_above == va)]
            if len(sub):
                print_row(row(sub, f"{g:14} & {tag}"))
        print()

    print("=== marginal: volume_ratio[L2] alone (any position) ===")
    print_row(row(df[df.vol_above], f"vol>{VOL_L2_THRESHOLD}"))
    print_row(row(df[~df.vol_above], f"vol<={VOL_L2_THRESHOLD}"))

    print("\n=== the combo actually being asked about: bottom_third AND vol_L2>1.0 ===")
    combo = df[(df.terc_L2 == "bottom_third") & (df.vol_above)]
    print_row(row(combo, "bottom_third & vol>1.0"))
    print("  (compare: bottom_third alone, any volume: n=249, hit=54.6%, net=+0.591%, t=+0.90)")


if __name__ == "__main__":
    main()
