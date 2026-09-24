"""Fresh-cross-section check of the two-touch-low rebound_retest rule
(locked params, retest_gap=2, market-drawdown exit DD_PCT=0.01) on a
CANDIDATE pool of 100 additional large-cap tickers, proposed for the live
universe to raise trade frequency (currently ~51 trades/yr pooled across
the existing 101-ticker LIVE_TICKERS). None of these 100 names were
touched while tuning params, retest_gap, or the drawdown threshold -- this
is the same "fresh cross-section" check as two_touch_low_dd_exit_oos.py's
81-ticker set, applied to a disjoint, never-before-seen pool per
validation-first-quant-work: a candidate must clear a fresh cross-section
before being added to the live signal, not just be assumed to replicate.

Pool selection: 100 S&P 100/500 large- or mega-cap names (all multi-$B
market cap, no penny stocks), chosen to be sector-diverse and disjoint
from LIVE_TICKERS (config.ticker_list + config.oos_ticker_list). Picked
by hand from familiar large-cap names across tech, healthcare, financials,
industrials, energy, materials, consumer, communications, utilities, and
real estate -- not itself a validation set, just a candidate pool; the
backtest below is what decides whether it's added.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_new100_check.py [dd_pct]
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
from daybyday_variants import summ
from two_touch_low_daybyday import PARAMS
from two_touch_low_daybyday_dd_exit import run_ticker

NEW_100 = [
    # tech / semis / software
    'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'PANW', 'CRWD', 'FTNT', 'ANET', 'MSI',
    'HPQ', 'DELL', 'NXPI', 'MCHP', 'ON', 'WDC', 'STX', 'TEAM', 'WDAY', 'DDOG',
    # consumer staples / discretionary
    'MDLZ', 'KHC', 'HSY', 'KDP', 'STZ', 'MNST', 'CLX', 'CHD', 'KR', 'SYY',
    'DG', 'DLTR', 'ROST', 'BBY', 'AZO', 'ULTA', 'EBAY', 'ETSY', 'LULU', 'DPZ',
    'CMG', 'MAR', 'HLT', 'BKNG', 'EXPE', 'CCL', 'RCL', 'NCLH',
    # healthcare
    'CI', 'HUM', 'ELV', 'CNC', 'MCK', 'COR', 'ZBH', 'BSX', 'EW', 'HCA',
    'DXCM', 'IDXX', 'IQV', 'A', 'MTD', 'WAT', 'RMD', 'ALGN', 'VRTX', 'REGN',
    'BIIB', 'MRNA',
    # financials
    'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'PGR', 'AFL', 'CB', 'AON', 'MMC',
    'MSCI', 'SPGI', 'MCO', 'ICE', 'CME', 'NDAQ', 'COF', 'DFS', 'SYF', 'TFC',
    'FITB',
    # industrials
    'GE', 'ETN', 'PH', 'ROK', 'CMI', 'PCAR', 'FAST', 'WM', 'NSC',
]
assert len(NEW_100) == 100, len(NEW_100)
assert len(set(NEW_100) & set(LIVE_TICKERS)) == 0, set(NEW_100) & set(LIVE_TICKERS)


def main():
    dd_pct = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    print(f"candidate pool, n={len(NEW_100)} (disjoint from the live 101-ticker universe, "
          f"never touched while tuning params, retest_gap, or dd_pct={dd_pct:.0%})\n", flush=True)

    series, failed = pull_all(NEW_100 + ["SPY"])
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, volume = {}, {}
    for t in NEW_100:
        if t not in series:
            continue
        c, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)
    spy_close, _ = series["SPY"]
    spy_close = spy_close.copy(); spy_close.index = pd.to_datetime(spy_close.index)

    for pct, label in [(1.0, "no early exit"), (dd_pct, f"market-dd exit {dd_pct:.0%}")]:
        args = [(t, close[t], volume[t], spy_close.reindex(close[t].index, method="ffill"), PARAMS, pct)
                for t in close]
        with Pool(min(len(close), cpu_count())) as pool:
            out = pool.map(run_ticker, args)
        df = pd.DataFrame([row for o in out for row in o])
        if df.empty:
            print(f"  {label}: no trades")
            continue
        df = attach_benchmark(df, ew)
        print(summ(df, label))
        if pct < 1.0:
            print(f"  exit reasons: {df.exit_reason.value_counts().to_dict()}")
            df.to_csv(os.path.join(HERE, "two_touch_low_new100_trades.csv"), index=False)


if __name__ == "__main__":
    main()
