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
  4. everything else ("watch"), split into two groups for display ordering
     only: supports with AGE <= DISPLAY_AGE_CUTOFF_DAYS sort above supports
     with AGE > DISPLAY_AGE_CUTOFF_DAYS. This grouping is purely cosmetic and
     is separate from MAX_AGE_DAYS, the live rule's actual age window (see
     "zone, aged out" below, and tech_levels_notes.md's level-age finding).
     Within each group, rows are ordered by DIST ascending, closest to firing
     first. A ticker with no support currently tracked sorts last.

Every BUY/HELD/SOLD row also gets a `bounce_1d=...` tag in NOTE -- the
next-day return after the triggering level's birth date (logged by
tech_level_continuation_live.py's bounce_1d_after_birth, purely diagnostic,
never used in the buy/sell decision, carried through the position's life
once opened). Tagged "small"/"large" against BOUNCE_REF_PCT, an
experimental reference point (not a validated threshold):
tech_levels_notes.md's 2026-09-19 "post-birth bounce" section found a
SMALLER post-birth bounce empirically preceded better subsequent trades on
both ticker_list and oos, but also that gating on it would have cut total
realized CAGR roughly in half at every capital-slot count tested (the
discarded half was still solidly profitable) -- shown here for reference
only, not as a signal to act on.

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

  --live  Read-only "what to buy right now" view: pulls current prices (the
          still-forming bar, so run it near the close), rebuilds levels, and
          shows live price vs. level per ticker. Writes nothing -- no log rows,
          no position changes. Held names are compared to the level they were
          bought on, so a price under that level is visible at a glance.
"""
import os
import sys

import pandas as pd

from tech_level_continuation_live import LIVE_TICKERS, MAX_AGE_DAYS, NEAR_PCT, HOLD_DAYS

SIGNAL_LOG_FILE = os.path.join(os.path.dirname(__file__), "data", "continuation_signal_log.csv")

# Display-only ordering cutoff: within the "watch" group, supports with age <=
# this many days sort above older ones. This is independent of MAX_AGE_DAYS
# (imported above), which governs actual buy-signal eligibility in the live
# script and must not be changed for display purposes.
DISPLAY_AGE_CUTOFF_DAYS = 10

# Experimental reference point only, not a validated threshold -- see
# tech_levels_notes.md, 2026-09-19 "post-birth bounce" section. Roughly the
# backtest median (0.73-0.74% on ticker_list/oos); used here purely to tag
# BUY/HELD rows "small"/"large" for reference, never to filter or decide.
# REMEMBER: SMALL is the good sign for a buy (a small post-birth bounce
# preceded better trades); LARGE is the unfavorable side.
BOUNCE_REF_PCT = 0.0075

STATUS_LABEL = {"buy": "BUY", "held": "HELD", "sell": "SOLD", "watch": "watch"}
STATUS_RANK = {"buy": 0, "held": 1, "sell": 2, "watch": 3}


def load_latest(path=SIGNAL_LOG_FILE):
    df = pd.read_csv(path, parse_dates=["run_date", "as_of"])
    return df.sort_values("run_date").groupby("ticker", as_index=False).tail(1).set_index("ticker")


def live_view():
    import datetime
    import socket
    from tech_level_live import pull_all, SOCKET_TIMEOUT
    from tech_level_naive_strategy import load_fixed_combo
    from tech_level_continuation_live import evaluate_ticker, load_positions

    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    combo = load_fixed_combo()
    series, failed = pull_all(LIVE_TICKERS)
    positions = load_positions()
    log = load_latest()
    # level each open position was bought on = its most recent 'buy' row
    buys = pd.read_csv(SIGNAL_LOG_FILE, parse_dates=["run_date", "as_of"])
    buys = buys[buys["event"] == "buy"].sort_values("run_date").groupby("ticker").tail(1).set_index("ticker")

    rows = []
    for ticker in LIVE_TICKERS:
        if ticker not in series:
            continue
        close, _ = series[ticker]
        price = float(close.iloc[-1])
        as_of = pd.Timestamp(close.index[-1]).date()
        # fresh evaluation with NO position state: is this a buy at this exact price?
        row, _ = evaluate_ticker(ticker, close, combo, {}, combo["distance"])
        held = ticker in positions
        lo, hi, age, birth = row["support_low"], row["support_high"], row["support_age_days"], row["support_birth"]
        if held and ticker in buys.index:
            b = buys.loc[ticker]
            lo, hi, age, birth = b["support_low"], b["support_high"], None, b["support_birth"]
        vs = (price - hi) / hi if pd.notna(hi) else None
        rows.append(dict(ticker=ticker, price=price, as_of=as_of, event=row["event"], held=held,
                         lo=lo, hi=hi, age=age, birth=birth, vs=vs,
                         prov=row["provisional"], entry=positions.get(ticker, {}).get("entry_price")))

    df = pd.DataFrame(rows)
    df["action"] = "watch"
    df.loc[df["event"] == "buy", "action"] = "BUY NOW"
    df.loc[df["held"], "action"] = "HELD"
    df["rank"] = df["action"].map({"BUY NOW": 0, "HELD": 1, "watch": 2})
    df["absvs"] = df["vs"].abs().fillna(float("inf"))
    df = df.sort_values(["rank", "absvs"])

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"LIVE watchlist -- {stamp} local, bar dated {df['as_of'].max()} (intraday, unsettled)\n"
          f"BUY NOW = live price 0-{NEAR_PCT:.0%} above an unbroken support born < {MAX_AGE_DAYS}d ago.\n"
          f"vs LEVEL = live price relative to the level's top edge (negative = UNDER the level).\n")
    header = f"{'':1} {'TICKER':6} {'ACTION':8} {'LIVE':>9} {'LEVEL':>15} {'vs LEVEL':>9} {'BORN':>11}  NOTE"
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        if r["action"] == "watch" and (pd.isna(r["vs"]) or r["vs"] > 0.03):
            continue  # only show names within 3% above a level
        level = f"{r['lo']:.2f}-{r['hi']:.2f}" if pd.notna(r["hi"]) else "--"
        vs = f"{r['vs']:+.1%}" if pd.notna(r["vs"]) else "--"
        note = ""
        if r["action"] == "HELD":
            note = f"entry {r['entry']:.2f} ({r['price'] / r['entry'] - 1:+.1%})"
            if pd.notna(r["vs"]) and r["vs"] < 0:
                note += " -- UNDER the level it was bought on"
        elif r["action"] == "BUY NOW" and r["prov"] is True:
            note = "provisional"
        print(f"{'*' if r['action'] == 'BUY NOW' else ' '} {r['ticker']:6} {r['action']:8} {r['price']:>9.2f} "
              f"{level:>15} {vs:>9} {str(r['birth'] or '--'):>11}  {note}")
    if failed:
        print(f"\nno data: {', '.join(sorted(failed))}")


def main():
    if "--live" in sys.argv[1:]:
        return live_view()
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
    latest["eligible"] = latest["support_age_days"] <= DISPLAY_AGE_CUTOFF_DAYS
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

        if row["event"] in ("buy", "held", "sell") and pd.notna(row.get("bounce_1d")):
            tag = ("small = favorable" if row["bounce_1d"] < BOUNCE_REF_PCT
                   else "large = unfavorable")
            bounce_note = f"bounce_1d={row['bounce_1d']:+.1%} ({tag}, experimental)"
            note = f"{note} | {bounce_note}" if note else bounce_note

        print(f"{marker} {ticker:6} {status:6} {price:>9} {level:>15} {buy_band:>15} {dist:>7} {age:>5}  {note}")


if __name__ == "__main__":
    main()
