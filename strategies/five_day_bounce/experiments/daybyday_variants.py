"""Strict day-by-day (fully causal) test of level-definition variants.

For every trading day d, levels are rebuilt from close[:d] only -- exactly what
a live run on d could see, including single-bar-confirmed touches at the tail.
Entry, level age and the resistance used for the exit all come from that
truncated build; nothing is hindsight. Same frozen rule as the live strategy
(entry within 1% above an unbroken support younger than 5 calendar days,
5-session hold or exit on entering entry-time resistance, 10 bps cost, one
position per ticker), same fixed combo.

Variants (parameters fixed a priori, not tuned on this sample):
  base      -- current rule
  gap60     -- the two touches forming a level are <= 60 sessions apart
  conf2     -- level birth must be >= 2 sessions old at entry
  gap60+conf2

Read-only. Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/daybyday_variants.py FDX TMO MSFT GD PEP
"""
import datetime
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_levels import find_touches, Level, mark_broken
from tech_level_naive_strategy import load_fixed_combo, active_support_resistance, equal_weight_curve, attach_benchmark
from tech_level_live import pull_all

NEAR, HOLD, MAX_AGE, GAP, CONF, WARMUP = 0.01, 5, 5, 60, 2, 300
VARIANTS = {"base": (None, 0), "gap60": (GAP, 0), "conf2": (None, CONF), "gap60+conf2": (GAP, CONF)}


def build(close, touches, width, max_gap):
    pos = {d: i for i, d in enumerate(close.index)}
    levels, first, open_c = [], {}, []
    for date, price in touches:
        if any(l.band[0] <= price <= l.band[1] for l in levels):
            continue
        within = [c for c in open_c if abs(price - c[1]) < width * c[1]
                  and (max_gap is None or pos[date] - pos[c[0]] <= max_gap)]
        if within:
            m = min(within, key=lambda c: abs(price - c[1]))
            lvl = Level(band=(min(m[1], price), max(m[1], price)), birth_date=date)
            levels.append(lvl)
            first[id(lvl)] = pos[date] - pos[m[0]]
            open_c.remove(m)
        else:
            open_c.append((date, price))
    mark_broken(levels, close)
    return levels, first


def run_ticker(args):
    ticker, close, combo = args
    combo = dict(combo)
    width = combo.pop("tech_width")
    for k in ("consider_volume", "volume_height", "volume_prominence"):
        combo.pop(k, None)
    idx, n = close.index, len(close)
    busy = {v: -1 for v in VARIANTS}
    trades = []
    for i in range(WARMUP, n - HOLD):
        if all(busy[v] >= i for v in VARIANTS):
            continue
        trunc = close.iloc[:i + 1]
        touches = find_touches(trunc, **combo)
        built = {g: build(trunc, touches, width, g) for g in {gap for gap, _ in VARIANTS.values()}}
        date, price = idx[i], float(close.iloc[i])
        for v, (gap, conf) in VARIANTS.items():
            if busy[v] >= i:
                continue
            levels, first = built[gap]
            sup, res = active_support_resistance(levels, date, price)
            if sup is None:
                continue
            dist = (price - sup.band[1]) / price
            age_days = (date - pd.Timestamp(sup.birth_date)).days
            age_sess = i - idx.get_loc(pd.Timestamp(sup.birth_date))
            if not (0 <= dist <= NEAR and age_days < MAX_AGE and age_sess >= conf):
                continue
            j_exit, reason = i + HOLD, "hold_days"
            for j in range(i + 1, i + HOLD + 1):
                if res is not None and res.band[0] <= close.iloc[j] <= res.band[1]:
                    j_exit, reason = j, "resistance"
                    break
            ret = float(close.iloc[j_exit] / price - 1)
            # gap between the level's two touches, measured on the unrestricted build
            g_base = built[None][1].get(id(sup)) if gap is None else first.get(id(sup))
            trades.append(dict(variant=v, stock=ticker, entry_date=date, exit_date=idx[j_exit], exit_reason=reason,
                               ret=ret, ret_net=ret - 10 / 1e4, touch_gap=g_base, provisional_sessions=age_sess))
            busy[v] = j_exit
    return trades


def summ(df, label):
    n = len(df)
    if n < 3:
        return f"  {label:22} n={n}"
    ex = df.excess_ret
    return (f"  {label:22} n={n:4d}  hit={(df.ret > 0).mean():5.1%}  net={df.ret_net.mean():+.3%}  "
            f"excess={ex.mean():+.3%}  t={ex.mean() / ex.std() * np.sqrt(n):+.2f}")


def main():
    tickers = sys.argv[1:] or ["FDX", "TMO", "MSFT", "GD", "PEP"]
    series, failed = pull_all(tickers)
    today = datetime.date.today()
    close = {}
    for t in tickers:
        c = series[t][0].copy()
        c.index = pd.to_datetime(c.index)
        close[t] = c[c.index.date < today]  # drop today's still-forming bar
    ew = equal_weight_curve(close)
    combo = load_fixed_combo()
    with Pool(len(tickers)) as p:
        out = p.map(run_ticker, [(t, close[t], combo) for t in tickers])
    df = attach_benchmark(pd.DataFrame([t for o in out for t in o]), ew)
    df.to_csv(os.path.join(HERE, "daybyday_variants_trades.csv"), index=False)
    print(f"\nDay-by-day causal test, {', '.join(tickers)}; {min(len(c) for c in close.values())}+ sessions each\n")
    for v in VARIANTS:
        print(summ(df[df.variant == v], v))
    b = df[df.variant == "base"]
    print("\nbase trades split by the gap between the level's two touches:")
    print(summ(b[b.touch_gap <= GAP], f"gap <= {GAP} sessions"))
    print(summ(b[b.touch_gap > GAP], f"gap > {GAP} sessions (stale)"))
    print("\nper ticker (base):")
    for t in tickers:
        print(summ(b[b.stock == t], t))


if __name__ == "__main__":
    main()
