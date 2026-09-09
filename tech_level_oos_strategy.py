"""Out-of-sample confirmation of the naive support/resistance strategy
(tech_level_naive_strategy.py) on config.oos_ticker_list.

Those 60 names are disjoint from config.ticker_list and have never
contributed to DEFAULT_GRID's universe calibration (tech_level_search.py) or
to picking the fixed combo (distance=10, prominence=0.01, tech_width=0.008
in data/live_combos.json) -- so this is a genuinely fresh cross-section for
the exact rule already validated on ticker_list: buy within 1% of a support
band, sell at hold_days or on touching the nearest known resistance,
whichever comes first. Same combo, same entry/exit rule, same placebo
control as tech_level_naive_strategy.py -- only the universe changes. If the
edge disappears here, the ticker_list result was riding on DEFAULT_GRID
having been calibrated around exactly those 40 names' peak-prominence
distribution, not a discovered rule.

NVDA is dropped from the 60 despite being listed in config.oos_ticker_list:
data/tech_levels/NVDA.csv already holds a full 84-combo grid-search touch
log (all combos present, dated 2026-07-20), even though tech_levels_notes.md
says NVDA was deliberately excluded from grid search because it's reserved.
That log predates the note, so someone ran it before the reservation was
written down. It doesn't bias the fixed combo tested here (that combo's
parameters came from ticker_list's calibration, not from NVDA's grid
results), but NVDA is no longer a name nobody has inspected, so it's
excluded rather than silently included.

Loads prices from the frozen data/oos/*.csv pull (config.oos) -- the same
pre-pulled snapshot vol_regime.py/volume_model.py/alpha.py already use for
this universe -- rather than a fresh yfinance call, so the record doesn't
depend on when this script happens to run.

Run: ./venv/bin/python tech_level_oos_strategy.py
"""
import pandas as pd

import config
from stock_class import StockList
from tech_level_naive_strategy import load_fixed_combo, run_backtest

EXCLUDE = {"NVDA"}  # see module docstring


def main():
    combo = load_fixed_combo()
    tickers = [t for t in config.oos_ticker_list if t not in EXCLUDE]

    print(f"loading {len(tickers)} OOS tickers from {config.oos} ...")
    sl = StockList(tickers)
    sl.load_data(config.oos)

    series = {}
    for t in tickers:
        try:
            c = sl[t].data["close"].dropna()
            c.index = pd.to_datetime(c.index)
            if not c.empty:
                series[t] = c
        except Exception:
            continue
    print(f"got price series for {len(series)} tickers")

    run_backtest(series, combo, "data/oos_strategy_trades.csv", label=" (OOS universe)")


if __name__ == "__main__":
    main()
