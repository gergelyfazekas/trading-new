"""Day-by-day (fully causal) test: a level = two troughs less than 10 sessions
apart (within tech_width of each other), not yet broken.

Under the fixed combo (find_peaks distance=10) two detected troughs are always
>= 10 sessions apart, so this variant lowers find_peaks distance to 1 (any
local low with >= 1% log prominence) and enforces the <10-session gap itself.
Peaks are ignored entirely. Levels are rebuilt for each day from close[:d]
only; entry rule, 5-session hold / resistance exit and 10 bps cost are the
frozen live rule's. Resistance for the exit comes from the variant's own
(trough-only) level set.

Variants:
  base        -- current live rule (any pair, distance=10), level age < 5d
  trough9     -- trough pairs <= 9 sessions apart, age < 5d
  trough9_any -- same levels, no age limit (any unbroken level)
Each reports the frozen exit and a plain 5-session hold (ret_hold5).

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/trough_pairs_daybyday.py [seed]
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
from tech_levels import find_touches
from tech_level_naive_strategy import load_fixed_combo, active_support_resistance, equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from daybyday_variants import build, summ, NEAR, HOLD, MAX_AGE, WARMUP

TROUGH_GAP = 9
VARIANTS = {"base": ("any", None, MAX_AGE), "trough9": ("trough", TROUGH_GAP, MAX_AGE),
            "trough9_any": ("trough", TROUGH_GAP, None)}


def run_ticker(args):
    ticker, close, combo = args
    combo = dict(combo)
    width = combo.pop("tech_width")
    for k in ("consider_volume", "volume_height", "volume_prominence"):
        combo.pop(k, None)
    trough_kw = dict(combo, distance=1)
    idx, n = close.index, len(close)
    busy = {v: -1 for v in VARIANTS}
    trades = []
    for i in range(WARMUP, n - HOLD):
        if all(busy[v] >= i for v in VARIANTS):
            continue
        trunc = close.iloc[:i + 1]
        date, price = idx[i], float(close.iloc[i])
        cache = {}
        for v, (mode, gap, max_age) in VARIANTS.items():
            if busy[v] >= i:
                continue
            key = (mode, gap)
            if key not in cache:
                if mode == "any":
                    touches = find_touches(trunc, **combo)
                else:
                    touches = [(d, p) for d, p, k in find_touches(trunc, with_kind=True, **trough_kw) if k == "trough"]
                cache[key] = build(trunc, touches, width, gap)[0]
            sup, res = active_support_resistance(cache[key], date, price)
            if sup is None:
                continue
            dist = (price - sup.band[1]) / price
            age_days = (date - pd.Timestamp(sup.birth_date)).days
            if not (0 <= dist <= NEAR and (max_age is None or age_days < max_age)):
                continue
            j_exit, reason = i + HOLD, "hold_days"
            for j in range(i + 1, i + HOLD + 1):
                if res is not None and res.band[0] <= close.iloc[j] <= res.band[1]:
                    j_exit, reason = j, "resistance"
                    break
            ret = float(close.iloc[j_exit] / price - 1)
            ret5 = float(close.iloc[i + HOLD] / price - 1)
            trades.append(dict(variant=v, stock=ticker, entry_date=date, exit_date=idx[j_exit], exit_reason=reason,
                               ret=ret, ret_net=ret - 10 / 1e4, ret_hold5=ret5 - 10 / 1e4, age_days=age_days))
            busy[v] = j_exit
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    tickers = random.Random(seed).sample(LIVE_TICKERS, 10)
    print(f"seed {seed}: {', '.join(tickers)}")
    series, failed = pull_all(tickers)
    today = datetime.date.today()
    close = {}
    for t in tickers:
        if t in series:
            c = series[t][0].copy()
            c.index = pd.to_datetime(c.index)
            close[t] = c[c.index.date < today]
    ew = equal_weight_curve(close)
    combo = load_fixed_combo()
    with Pool(min(len(close), cpu_count())) as p:
        out = p.map(run_ticker, [(t, close[t], combo) for t in close])
    df = pd.DataFrame([t for o in out for t in o])
    df = attach_benchmark(df, ew)
    df["excess_hold5"] = df.ret_hold5 - df.bench_ret
    df.to_csv(os.path.join(HERE, "trough_pairs_daybyday_trades.csv"), index=False)
    print(f"\nDay-by-day causal test, {len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print("frozen exit (5 sessions or resistance):")
    for v in VARIANTS:
        print(summ(df[df.variant == v], v))
    print("\nplain 5-session hold:")
    alt = df.assign(ret_net=df.ret_hold5, excess_ret=df.excess_hold5, ret=df.ret_hold5 + 10 / 1e4)
    for v in VARIANTS:
        print(summ(alt[alt.variant == v], v))
    print("\nper ticker (trough9, frozen exit):")
    for t in close:
        print(summ(df[(df.variant == "trough9") & (df.stock == t)], t))


if __name__ == "__main__":
    main()
