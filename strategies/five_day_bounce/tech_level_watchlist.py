"""Read-only viewer for tech_level_continuation_live.py's signal log.

Does no pulling and no level computation -- purely reads the latest logged
row per ticker from data/continuation_signal_log.csv and prints all 101 names
from LIVE_TICKERS (config.ticker_list union config.oos_ticker_list) as one
sorted table:

  1. BUY  -- the live rule actually fired this on its latest run (event ==
     'buy' in the log, exactly what tech_level_continuation_live.py decided --
     this viewer never re-derives or overrides that call). Always on top.
     A [P] suffix on STATUS means the triggering level is still provisional
     (age in *sessions* <= distance=10) -- see that script's module docstring:
     later price action could still un-confirm the touch that created it.
  2. HELD -- currently in an open position (bought on an earlier run, not yet
     sold). NOTE shows entry price, unrealized return, days held out of
     HOLD_DAYS, and the resistance band that triggers an early exit if hit
     (blank if no resistance was active at entry -- exits on HOLD_DAYS only).
  3. SOLD -- exited on the latest run. NOTE shows entry/exit price, realized
     return, days held, and why (hold_days or resistance).
  4. everything else ("watch"), split into two groups: supports with
     AGE < MAX_AGE_DAYS (still inside the rule's age window -- can still fire
     on a future run if price closes into BUY BAND) sort above supports with
     AGE >= MAX_AGE_DAYS (aged out for good; that specific level can never
     fire again, see tech_levels_notes.md's level-age finding). Within each
     group, rows are ordered by DIST ascending, closest to firing first.
     A ticker with no support currently tracked sorts last.

A "watch" row whose price already sits inside BUY BAND but is aged out gets a
"zone, aged out" NOTE instead of being shown as BUY -- that combination is
exactly the levels the age-bucket sensitivity sweep found have ~zero (OOS) or
negative (ticker_list) excess return, not the validated <5-day effect (see
tech_levels_notes.md, "Level age at entry", and NOTES.md in this folder). It's
worth seeing -- it explains why a name sitting right at a familiar level isn't
a signal -- but it must never be relabeled BUY. This was a real bug on
2026-09-09 (GILD shown as BUY at 11 days old); fixed by making STATUS a direct
read of the log's own `event`, never a recomputation.

The DIST and AGE columns are shown side by side deliberately: a "watch" row
close in price (small DIST) but old in age (AGE >= MAX_AGE_DAYS) is one whose
support already aged out of eligibility -- distinct from one that's simply too
far away in price. Reading both columns together tells you which is which; a
single derived flag would hide it.

BUY BAND is (support_high, support_high + 1%) -- the price range the close
actually has to land in for that ticker to be considered a buy event. Not the
full support band: a close inside [support_low, support_high] does NOT buy
under the live rule (active_support_resistance requires strictly above), and
closing inside was checked and found to be a materially worse entry, not a
stronger one -- see tech_levels_notes.md, 2026-09-08 section.

Run (from repo root): ./venv/bin/python strategies/five_day_bounce/tech_level_watchlist.py
"""
import os

import pandas as pd

from tech_level_continuation_live import LIVE_TICKERS, MAX_AGE_DAYS, NEAR_PCT, HOLD_DAYS

SIGNAL_LOG_FILE = os.path.join(os.path.dirname(__file__), "data", "continuation_signal_log.csv")

STATUS_LABEL = {"buy": "BUY", "held": "HELD", "sell": "SOLD", "watch": "watch"}
STATUS_RANK = {"buy": 0, "held": 1, "sell": 2, "watch": 3}


def load_latest(path=SIGNAL_LOG_FILE):
    df = pd.read_csv(path, parse_dates=["run_date", "as_of"])
    return df.sort_values("run_date").groupby("ticker", as_index=False).tail(1).set_index("ticker")


def main():
    latest = load_latest()
    latest = latest.reindex(LIVE_TICKERS)

    latest["dist"] = (latest["price"] - latest["support_high"]) / latest["price"]
    latest["dist"] = latest["dist"].fillna(float("inf"))
    latest["buy_band_low"] = latest["support_high"]
    latest["buy_band_high"] = latest["support_high"] * 1.01
    in_buy_band = latest["price"].between(latest["buy_band_low"], latest["buy_band_high"])
    aged_out = latest["support_age_days"] >= MAX_AGE_DAYS
    latest["note"] = ""
    latest.loc[(latest["event"] == "watch") & in_buy_band & aged_out, "note"] = "zone, aged out"
    latest.loc[(latest["event"] == "buy") & (latest["provisional"] == True), "note"] = "provisional"  # noqa: E712

    latest["rank"] = latest["event"].map(STATUS_RANK).fillna(3)
    latest["eligible"] = latest["support_age_days"] < MAX_AGE_DAYS
    latest = latest.sort_values(
        ["rank", "eligible", "dist"], ascending=[True, False, True]
    )

    as_of = latest["as_of"].dropna().max()
    run_date = latest["run_date"].dropna().max()
    as_of_str = as_of.date() if pd.notna(as_of) else "?"
    run_date_str = run_date.date() if pd.notna(run_date) else "?"

    header = (f"{'':1} {'TICKER':6} {'STATUS':6} {'PRICE':>9} {'LEVEL':>15} "
              f"{'BUY BAND':>15} {'DIST':>7} {'AGE':>5}  {'NOTE'}")
    print(f"Continuation watchlist -- prices as of {as_of_str} (logged {run_date_str}); "
          f"near_pct={NEAR_PCT:.0%}, max_age={MAX_AGE_DAYS}d\n")
    print(header)
    print("-" * len(header))

    for ticker, row in latest.iterrows():
        if pd.isna(row.get("event")):
            print(f"  {ticker:6} {'--':6} {'--':>9} {'--':>15} {'--':>15} {'--':>7} {'--':>5}")
            continue

        status = STATUS_LABEL.get(row["event"], row["event"])
        if row["event"] == "buy" and row.get("provisional") is True:
            status += "[P]"
        price = f"{row['price']:.2f}"
        if pd.notna(row.get("support_low")):
            level = f"{row['support_low']:.2f}-{row['support_high']:.2f}"
            buy_band = f"{row['buy_band_low']:.2f}-{row['buy_band_high']:.2f}"
        else:
            level = "--"
            buy_band = "--"
        dist = f"{row['dist']:.1%}" if row["dist"] != float("inf") else "--"
        age = f"{int(row['support_age_days'])}d" if pd.notna(row.get("support_age_days")) else "--"
        marker = "*" if row["event"] == "buy" else " "
        note = row["note"] if row["note"] else ""
        if row["event"] == "held":
            unrl = row["price"] / row["entry_price"] - 1.0
            note = f"entry {row['entry_price']:.2f} ({unrl:+.1%} unrl), day {int(row['days_held'])}/{HOLD_DAYS}"
            if pd.notna(row.get("resist_low")):
                note += f", exits early if price re-enters {row['resist_low']:.2f}-{row['resist_high']:.2f}"
        elif row["event"] == "sell":
            note = (f"entry {row['entry_price']:.2f} -> exit {row['exit_price']:.2f} "
                     f"({row['ret']:+.1%}), held {int(row['days_held'])}d ({row['reason']})")
        print(f"{marker} {ticker:6} {status:6} {price:>9} {level:>15} {buy_band:>15} {dist:>7} {age:>5}  {note}")


if __name__ == "__main__":
    main()
