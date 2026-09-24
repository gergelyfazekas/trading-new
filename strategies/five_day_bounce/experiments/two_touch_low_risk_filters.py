"""Two risk overlays on top of the locked two_touch_low rule
(rebound_retest branch, retest_gap=2), tested separately and combined, on
the full 101-ticker LIVE_TICKERS universe. Motivated by
two_touch_low_extremes_chart.py's finding: the biggest winners and losers
are indistinguishable at entry -- both are quiet, ordinary-looking
setups -- and the extremes are driven by news/macro events landing in the
5-day hold, not by anything in the pattern itself.

1. EARNINGS FILTER -- skip a candidate if that ticker has an earnings
   reaction day (earnings.py's reaction_day, already-built cache of true
   announcement dates) anywhere in [buy_day, exit_day]. A priori, no
   tuning: any earnings date inside the hold window disqualifies the
   trade, full stop.

2. MARKET-DRAWDOWN EARLY EXIT -- exit at the first day within the hold
   where SPY's cumulative return since entry drops below
   -market_dd_pct (tested at 1%, 2%, 3%, all three a priori, none tuned
   on this sample). Otherwise the exit is the usual 5-session hold.

decide() (the entry rule) is untouched and reused as-is -- these overlays
only change which candidates are accepted and when a position exits, not
whether a low/rebound/retest pattern is recognized.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_risk_filters.py [seed] [n_tickers]
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
from daybyday_variants import summ
from two_touch_low_daybyday import PARAMS, HOLD, COST, WARMUP, decide
from earnings import load_earnings, reaction_day

RETEST_GAP = 2
DD_THRESHOLDS = [0.01, 0.02, 0.03]


def earnings_positions(close, earn_ticker_rows):
    """Set of integer positions in `close.index` where an earnings
    reaction lands, for one ticker."""
    pos = set()
    for _, row in earn_ticker_rows.iterrows():
        p = reaction_day(close.index, row.date, row.hour)
        if p is not None:
            pos.add(p)
    return pos


def run_ticker(args):
    ticker, close, volume, spy_ret, earn_pos, p, dd_thresholds = args
    idx, n = close.index, len(close)
    busy = -1
    # one trade list per config: base, earnings-filtered, and one per dd threshold
    configs = ["base", "earn"] + [f"dd{int(t*100)}" for t in dd_thresholds] + \
              [f"earn+dd{int(t*100)}" for t in dd_thresholds]
    trades = {c: [] for c in configs}
    busy_by = {c: -1 for c in configs}

    for L in range(WARMUP, n - HOLD - RETEST_GAP - 1):
        decision = decide(close, volume, L, p, retest_gap=RETEST_GAP)
        if decision is None or decision["branch"] != "rebound_retest":
            continue
        buy_day = L + decision["offset"]
        has_earnings = any(buy_day <= e <= buy_day + HOLD for e in earn_pos)

        # mechanical exit (base / dd-only configs)
        exit_i = buy_day + HOLD
        if exit_i >= n:
            continue
        c_entry = float(close.iloc[buy_day])
        ret = float(close.iloc[exit_i] / c_entry - 1)
        row = dict(stock=ticker, entry_date=idx[buy_day], exit_date=idx[exit_i],
                   ret=ret, ret_net=ret - COST, exit_reason="hold_days")

        # drawdown-based early exit -- scan forward from buy_day+1
        dd_exit = {}
        for thr in dd_thresholds:
            e_i, e_reason = exit_i, "hold_days"
            entry_spy = spy_ret.iloc[buy_day] if buy_day < len(spy_ret) else np.nan
            for j in range(buy_day + 1, min(buy_day + HOLD, n - 1) + 1):
                if j >= len(spy_ret) or pd.isna(entry_spy) or pd.isna(spy_ret.iloc[j]):
                    continue
                spy_since_entry = spy_ret.iloc[j] / entry_spy - 1
                if spy_since_entry <= -thr:
                    e_i, e_reason = j, "market_dd"
                    break
            r = float(close.iloc[e_i] / c_entry - 1)
            dd_exit[thr] = dict(stock=ticker, entry_date=idx[buy_day], exit_date=idx[e_i],
                                 ret=r, ret_net=r - COST, exit_reason=e_reason)

        if buy_day > busy_by["base"]:
            trades["base"].append(row)
            busy_by["base"] = exit_i
        if not has_earnings and buy_day > busy_by["earn"]:
            trades["earn"].append(row)
            busy_by["earn"] = exit_i
        for thr in dd_thresholds:
            key = f"dd{int(thr*100)}"
            r = dd_exit[thr]
            if buy_day > busy_by[key]:
                trades[key].append(r)
                busy_by[key] = idx.get_loc(r["exit_date"])
            ekey = f"earn+dd{int(thr*100)}"
            if not has_earnings and buy_day > busy_by[ekey]:
                trades[ekey].append(r)
                busy_by[ekey] = idx.get_loc(r["exit_date"])
    return trades


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    k = int(sys.argv[2]) if len(sys.argv) > 2 else len(LIVE_TICKERS)
    tickers = LIVE_TICKERS if k >= len(LIVE_TICKERS) else random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"{len(tickers)} tickers\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"])
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, volume = {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)

    spy_close, _ = series["SPY"]
    spy_close = spy_close.copy()
    spy_close.index = pd.to_datetime(spy_close.index)

    earn = load_earnings(tickers)
    earn_pos_by_ticker = {}
    for t in close:
        rows = earn[earn.stock == t]
        spy_aligned = spy_close.reindex(close[t].index, method="ffill")
        earn_pos_by_ticker[t] = earnings_positions(close[t], rows)

    args = []
    for t in close:
        spy_aligned = spy_close.reindex(close[t].index, method="ffill")
        args.append((t, close[t], volume[t], spy_aligned, earn_pos_by_ticker[t], PARAMS, DD_THRESHOLDS))

    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)

    configs = ["base", "earn"] + [f"dd{int(t*100)}" for t in DD_THRESHOLDS] + \
              [f"earn+dd{int(t*100)}" for t in DD_THRESHOLDS]
    for cfg in configs:
        rows = [row for o in out for row in o[cfg]]
        df = pd.DataFrame(rows)
        if df.empty:
            print(f"  {cfg:12} no trades")
            continue
        df = attach_benchmark(df, ew)
        print(summ(df, cfg))
        df.to_csv(os.path.join(HERE, f"two_touch_low_risk_{cfg}_trades.csv"), index=False)


if __name__ == "__main__":
    main()
