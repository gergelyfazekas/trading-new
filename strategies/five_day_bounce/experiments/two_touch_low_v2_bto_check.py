"""Diagnostic follow-up on two_touch_low_v2_charts.py's charts: several
losers had an L2 day where close sat right on top of the day's low --
not a buyers_took_over day -- suggesting those trades only got into the
combined-best combo via the volume_ratio[L2] branch of the OR gate, not
the candle condition. Checks that directly first (breaking down the
current combined-best combo's own trades by which branch of the L2
OR-gate actually fired), then reruns the rule with the OR replaced by
buyers_took_over alone.

The BTO-only variant is implemented by setting both volume thresholds to
+inf rather than touching decide()'s gate logic -- vr > inf is never true
for a finite ratio, so `bto or (vr > threshold)` collapses to `bto` with
zero risk of the variant silently drifting from the real gate code.
n_back, n_fwd and close_tolerance stay at their swept values throughout.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_bto_check.py [seed] [k]
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
from two_touch_low_v2_daybyday import run_ticker, buyers_took_over, volume_ratio

BASE_PARAMS = dict(
    n_back=5,
    n_fwd=3,
    volume_threshold_L=1.0,
    volume_threshold_L2=0.8,
    close_tolerance=0.005,
)
BTO_ONLY_PARAMS = dict(BASE_PARAMS, volume_threshold_L=float("inf"), volume_threshold_L2=float("inf"))
DD_PCT = 0.01


def row(df, label):
    if df.empty or len(df) < 3:
        return dict(label=label, n=len(df), hit=np.nan, net=np.nan, excess=np.nan, t=np.nan)
    ex = df.excess_ret
    return dict(label=label, n=len(df), hit=(df.ret > 0).mean(), net=df.ret_net.mean(),
                excess=ex.mean(), t=ex.mean() / ex.std() * np.sqrt(len(df)))


def print_row(r):
    print(f"  {r['label']:24} n={r['n']:4d}  hit={r['hit']:.1%}  net={r['net']:+.3%}  "
          f"excess={r['excess']:+.3%}  t={r['t']:+.2f}")


def gate_reason(bto, vr, threshold):
    vol_ok = not np.isnan(vr) and vr > threshold
    if bto and vol_ok:
        return "both"
    if bto:
        return "bto_only"
    if vol_ok:
        return "volume_only"
    return "neither(bug)"


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

    def run(params):
        args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], params, DD_PCT) for t in close]
        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, args)
        df = pd.DataFrame([r for o in out for r in o])
        if df.empty:
            return df
        return attach_benchmark(df, ew)

    base_df = run(BASE_PARAMS)
    print("=== current combined-best (OR: buyers_took_over OR volume) ===")
    print_row(row(base_df, "OR gate (baseline)"))

    reasons = []
    for _, tr in base_df.iterrows():
        t = tr.stock
        L2 = close[t].index.get_loc(pd.Timestamp(tr.entry_date))
        bto = buyers_took_over(high[t], low[t], close[t], L2)
        vr = volume_ratio(volume[t], L2)
        reasons.append(gate_reason(bto, vr, BASE_PARAMS["volume_threshold_L2"]))
    base_df = base_df.assign(l2_gate=reasons)
    print("\nbreakdown of the OR-gate baseline's own trades, by which L2 branch actually fired:")
    for g, sub in base_df.groupby("l2_gate"):
        print_row(row(sub, g))

    print("\n=== BTO-only variant (volume thresholds disabled, n_back/n_fwd/tolerance unchanged) ===")
    bto_df = run(BTO_ONLY_PARAMS)
    print_row(row(bto_df, "BTO-only gate"))


if __name__ == "__main__":
    main()
