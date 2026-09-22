"""Day-by-day (fully causal) test of the ORIGINAL frozen rule behind the watchlist.

Rule, exactly as tech_level_continuation_live.evaluate_ticker: fixed combo
(distance=10, prominence=0.01, tech_width=0.008); on each day d levels are
rebuilt from close[:d] only; buy if the nearest unbroken support is <5 calendar
days old and close is 0-1% above its band top; exit after 5 sessions or the
first close inside the entry-time resistance band; 10 bps cost; one position
per ticker. Also runs the whole-history (hindsight) backtest on the same
tickers for contrast, plus a matched-count random-entry placebo and splits by
the sessions between the level's two touches.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/original_rule_daybyday.py [seed] [n_tickers]
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
from tech_level_naive_strategy import (load_fixed_combo, build_levels, active_support_resistance,
                                       simulate, equal_weight_curve, attach_benchmark)
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from daybyday_variants import build, summ, NEAR, HOLD, MAX_AGE, WARMUP


def run_ticker(args):
    ticker, close, combo = args
    c = dict(combo)
    width = c.pop("tech_width")
    for k in ("consider_volume", "volume_height", "volume_prominence"):
        c.pop(k, None)
    idx, n = close.index, len(close)
    busy, trades, checked = -1, [], False
    for i in range(WARMUP, n - HOLD):
        if busy >= i:
            continue
        trunc = close.iloc[:i + 1]
        levels, first = build(trunc, find_touches(trunc, **c), width, None)
        if not checked:  # sanity: same level set as the real live builder
            ref = build_levels(trunc, combo)
            assert sorted((l.band, str(l.birth_date)) for l in ref) == sorted((l.band, str(l.birth_date)) for l in levels)
            checked = True
        date, price = idx[i], float(close.iloc[i])
        sup, res = active_support_resistance(levels, date, price)
        if sup is None:
            continue
        dist = (price - sup.band[1]) / price
        age_days = (date - pd.Timestamp(sup.birth_date)).days
        if not (0 <= dist <= NEAR and age_days < MAX_AGE):
            continue
        j_exit, reason = i + HOLD, "hold_days"
        for j in range(i + 1, i + HOLD + 1):
            if res is not None and res.band[0] <= close.iloc[j] <= res.band[1]:
                j_exit, reason = j, "resistance"
                break
        ret = float(close.iloc[j_exit] / price - 1)
        trades.append(dict(stock=ticker, entry_date=date, exit_date=idx[j_exit], exit_reason=reason,
                           ret=ret, ret_net=ret - 10 / 1e4, touch_gap=first.get(id(sup)), age_days=age_days))
        busy = j_exit
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    series, _ = pull_all(tickers)
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
    df = attach_benchmark(pd.DataFrame([t for o in out for t in o]), ew)
    df.to_csv(os.path.join(HERE, "original_rule_daybyday_trades.csv"), index=False)

    # hindsight backtest, same tickers, same rule
    hs = []
    for t, c in close.items():
        hs += simulate(t, c, build_levels(c, combo), hold_days=HOLD, near_pct=NEAR)
    hs = attach_benchmark(pd.DataFrame(hs), ew)
    hs = hs[hs.support_age_days < MAX_AGE]

    # placebo: random entries, matched count per ticker
    rng = np.random.default_rng(0)
    rows = []
    for t, cnt in df.groupby("stock").size().items():
        c = close[t]
        for i in rng.choice(np.arange(WARMUP, len(c) - HOLD), size=cnt, replace=False):
            rows.append({"entry_date": c.index[i], "exit_date": c.index[i + HOLD],
                         "ret_net": c.iloc[i + HOLD] / c.iloc[i] - 1 - 10 / 1e4})
    pe = attach_benchmark(pd.DataFrame(rows), ew).excess_ret.mean()

    print(f"\n{len(close)} tickers, ~{min(len(c) for c in close.values())}+ sessions each\n")
    print(summ(hs, "hindsight backtest"))
    print(summ(df, "DAY-BY-DAY (causal)"))
    print(f"  {'placebo excess (matched n)':22} {pe:+.3%}   causal lift = {df.excess_ret.mean() - pe:+.3%}")
    print(f"\n  exits: {df.exit_reason.value_counts().to_dict()}")
    print("\ncausal, by sessions between the level's two touches:")
    print(summ(df[df.touch_gap <= 60], "gap <= 60"))
    print(summ(df[df.touch_gap > 60], "gap > 60 (stale)"))
    print("\ncausal, by year:")
    for y, g in df.groupby(pd.DatetimeIndex(df.entry_date).year):
        print(summ(g, str(y)))
    print("\ncausal, per ticker:")
    for t in close:
        print(summ(df[df.stock == t], t))


if __name__ == "__main__":
    main()
