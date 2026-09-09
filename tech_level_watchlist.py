"""Read-only viewer for tech_level_continuation_live.py's signal log.

Does no pulling and no level computation -- purely reads the latest logged
row per ticker from data/continuation_signal_log.csv and prints all 40 names
from config.ticker_list as one sorted table:

  1. BUY  -- fired on the latest run, OR the logged price already sits
     inside BUY BAND (see below) even though the live rule didn't fire it
     (e.g. support_age_days == MAX_AGE_DAYS cuts it off by one day) --
     always on top. Trades happen manually near close, so a name that's
     price-eligible right now is shown as BUY regardless of why the
     automated tracker called it "watch".
  2. HELD -- currently in an open position.
  3. SOLD -- exited on the latest run.
  4. everything else ("watch"), split into two groups: supports with
     AGE < 6 days ("can still fire today") sort above supports with
     AGE >= 6 days. Within each group, age plays no further part --
     rows are ordered purely by DIST ascending, closest to firing first.
     Trades happen before close on the same day, not off the next
     morning's automated "buy" print, so a support that can still fire
     soon belongs near the top regardless of its age within the group.
     A ticker with no support currently tracked sorts last.

The DIST and AGE columns are shown side by side deliberately: a "watch" row
close in price (small DIST) but old in age (AGE >= 5d) is one whose support
already aged out of eligibility and will never fire from that level again --
distinct from one that's simply too far away in price. Reading both columns
together tells you which is which; a single derived flag would hide it.

BUY BAND is (support_high, support_high + 1%) -- the price range the close
actually has to land in for that ticker to be considered a buy event. Not
the full support band: a close inside [support_low, support_high] does NOT
buy under the live rule (active_support_resistance requires strictly above),
and closing inside was checked and found to be a materially worse entry, not
a stronger one -- see tech_levels_notes.md, 2026-09-08 section.

Run: ./venv/bin/python tech_level_watchlist.py
"""
import pandas as pd

from tech_level_continuation_live import LIVE_TICKERS

SIGNAL_LOG_FILE = "data/continuation_signal_log.csv"

STATUS_LABEL = {"buy": "BUY", "held": "HELD", "sell": "SOLD", "watch": "watch"}
STATUS_RANK = {"buy": 0, "held": 1, "sell": 2, "watch": 3}
FIRES_TODAY_AGE_THRESHOLD = 6


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
    latest.loc[(latest["event"] == "watch") & in_buy_band, "event"] = "buy"
    latest["rank"] = latest["event"].map(STATUS_RANK).fillna(3)
    latest["fires_today"] = latest["support_age_days"] < FIRES_TODAY_AGE_THRESHOLD
    latest = latest.sort_values(
        ["rank", "fires_today", "dist"], ascending=[True, False, True]
    )

    as_of = latest["as_of"].dropna().max()
    run_date = latest["run_date"].dropna().max()
    as_of_str = as_of.date() if pd.notna(as_of) else "?"
    run_date_str = run_date.date() if pd.notna(run_date) else "?"

    header = (f"{'':1} {'TICKER':6} {'STATUS':6} {'PRICE':>9} {'LEVEL':>15} "
              f"{'BUY BAND':>15} {'DIST':>7} {'AGE':>5}")
    print(f"Continuation watchlist -- prices as of {as_of_str} (logged {run_date_str})\n")
    print(header)
    print("-" * len(header))

    for ticker, row in latest.iterrows():
        if pd.isna(row.get("event")):
            print(f"  {ticker:6} {'--':6} {'--':>9} {'--':>15} {'--':>15} {'--':>7} {'--':>5}")
            continue

        status = STATUS_LABEL.get(row["event"], row["event"])
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
        print(f"{marker} {ticker:6} {status:6} {price:>9} {level:>15} {buy_band:>15} {dist:>7} {age:>5}")


if __name__ == "__main__":
    main()
