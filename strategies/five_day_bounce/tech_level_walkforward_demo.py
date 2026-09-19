"""Walk-forward demo of the five-day-bounce continuation signal on one
randomly picked stock, with levels recomputed from scratch at every single
day using only data available up to that day -- i.e. exactly what a live
script would have seen, day by day, rather than the grid-search convention
of building levels once over the whole history.

Reuses tech_level_continuation_live.evaluate_ticker() unmodified as the
per-day decision step (same entry/exit rule, same fixed a-priori combo) so
this can't silently drift from the actual live logic -- the only thing this
script adds is the day-by-day truncation loop and the charting.

For every day evaluate_ticker() fires a 'buy', saves a line chart of the
previous 10 and next 10 trading days (from the real, full price series -- the
walk-forward truncation only governs signal generation, not what actually
happened afterward) with the buy marked and the triggering support band (and
active resistance, if any) drawn as horizontal lines.

Also writes one combined CSV per ticker (not one per chart) with every day of
every plotted window, long-format, keyed by `event_id` (shared with the
chart's filename stem) so a chart and its underlying numbers can always be
matched back up -- see EVENTS_CSV_SUFFIX. This is what a later "here's what I
noticed on chart X" impression should be checked against, rather than
re-pulling fresh data (which would silently drift as yfinance revises/adds
bars) or eyeballing the PNG.

Each row also carries that day's volume, its trailing 10-trading-day average
(causal -- the 10 days strictly before it, never including the day itself),
and the ratio of the two, plus is_birth_day (the level's confirming 2nd
touch) alongside the existing is_buy_day -- exploring whether the market
"notices" a level (elevated volume) on the day it's born and/or the day price
returns to trade it, vs. an ordinary day in the same window. main() prints a
plain descriptive summary (birth-day / buy-day ratio vs. the same windows'
other days) at the end of each run -- not a significance claim, just the
numbers, since n is one stock's worth of events.

Usage: ./venv/bin/python strategies/five_day_bounce/tech_level_walkforward_demo.py [TICKER]
Charts + events CSV saved to strategies/five_day_bounce/data/walkforward_charts/<TICKER>/.
"""
import datetime
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from tech_level_live import pull_series
from tech_level_naive_strategy import load_fixed_combo
from tech_level_continuation_live import evaluate_ticker, MAX_AGE_DAYS, NEAR_PCT, HOLD_DAYS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHART_DIR = os.path.join(DATA_DIR, "walkforward_charts")
WARMUP_DAYS = 30  # skip the first stretch where too little history exists to form any level
WINDOW = 10        # trading days shown before/after the buy
VOLUME_LOOKBACK = 10  # trading days for the trailing average volume baseline
EVENTS_CSV_SUFFIX = "_walkforward_events.csv"

EVENT_COLUMNS = [
    "event_id", "ticker", "chart_file", "buy_date", "date", "day_offset", "close",
    "is_buy_day", "is_birth_day",
    "volume", "avg_volume_10d", "volume_ratio",
    "support_low", "support_high", "support_birth", "support_age_days",
    "resist_low", "resist_high",
]


def window_bounds(n, i):
    return max(0, i - WINDOW), min(n - 1, i + WINDOW)


def trailing_avg_volume(volume, j, lookback=VOLUME_LOOKBACK):
    """Mean volume over the `lookback` trading days strictly before position
    j -- never includes day j itself, so this is safe to compare day j's own
    volume against without look-ahead."""
    seg = volume.iloc[max(0, j - lookback):j]
    return float(seg.mean()) if len(seg) and seg.notna().any() else float("nan")


def window_volume_stats(close, volume, i, signal_row):
    """One dict per day in this event's plotted window -- computed once and
    shared by plot_buy() and build_event_rows() so the chart's volume panel
    and the CSV can never show different numbers for the same day.
    """
    n = len(close)
    lo, hi = window_bounds(n, i)
    birth_date = signal_row["support_birth"]

    stats_rows = []
    for j in range(lo, hi + 1):
        date = close.index[j].date()
        vol = float(volume.iloc[j]) if volume is not None else float("nan")
        avg_vol = trailing_avg_volume(volume, j) if volume is not None else float("nan")
        ratio = vol / avg_vol if avg_vol not in (0.0, None) and not np.isnan(avg_vol) else float("nan")
        stats_rows.append({
            "j": j, "date": date,
            "is_buy_day": j == i,
            "is_birth_day": birth_date is not None and date == birth_date,
            "volume": vol, "avg_volume_10d": avg_vol, "volume_ratio": ratio,
        })
    return stats_rows


def run_walkforward(ticker, close, combo):
    """Day-by-day: truncate the close series to 'as of that day', hand it to
    the real live decision function, record every buy. Returns list of
    signal_row dicts (the 'buy' ones) plus their integer position in `close`.
    """
    distance = combo["distance"]
    positions = {}
    buys = []

    n = len(close)
    for i in range(WARMUP_DAYS, n):
        close_so_far = close.iloc[: i + 1]
        signal_row, _trade_row = evaluate_ticker(ticker, close_so_far, combo, positions, distance)
        if signal_row["event"] == "buy":
            buys.append((i, signal_row))

    return buys


def plot_buy(ticker, close, i, signal_row, vol_stats, out_path):
    """x-axis is trading-day position, not calendar date -- weekends (and any
    holiday gaps) have no bar/point to begin with, so plotting them on an
    actual date axis leaves a visible empty gap every 5 bars and makes the
    window look like it's broken into clusters. Using position 0..N-1 with
    date strings as tick labels shows the trading days back-to-back, which is
    what "previous/next 10 trading days" is supposed to mean anyway.
    """
    n = len(close)
    lo, hi = window_bounds(n, i)
    x = list(range(lo, hi + 1))
    labels = [close.index[j].strftime("%Y-%m-%d") for j in x]

    fig, (ax, vax) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )

    ax.plot(x, close.iloc[lo:hi + 1].values, color="#1f5fa8", linewidth=1.6,
             marker="o", markersize=3, label=ticker)

    buy_price = close.iloc[i]
    ax.scatter([i], [buy_price], color="#d62728", s=140, zorder=5, marker="*", label="buy signal")

    support_low, support_high = signal_row["support_low"], signal_row["support_high"]
    if support_low is not None:
        ax.axhline(support_low, color="#2ca02c", linestyle="--", linewidth=1.2)
        ax.axhline(support_high, color="#2ca02c", linestyle="--", linewidth=1.2,
                    label=f"support band [{support_low:.2f}, {support_high:.2f}]")
        ax.axhspan(support_low, support_high, color="#2ca02c", alpha=0.08)

    resist_low, resist_high = signal_row["resist_low"], signal_row["resist_high"]
    if resist_low is not None:
        ax.axhline(resist_low, color="#ff7f0e", linestyle=":", linewidth=1.2)
        ax.axhline(resist_high, color="#ff7f0e", linestyle=":", linewidth=1.2,
                    label=f"resistance band [{resist_low:.2f}, {resist_high:.2f}]")

    ax.set_ylabel("close")
    age = signal_row["support_age_days"]
    born = signal_row["support_birth"]
    ax.set_title(
        f"{ticker} -- buy signal {close.index[i].date()} @ {buy_price:.2f}\n"
        f"support born {born} ({age}d old, max age {MAX_AGE_DAYS}d) | "
        f"near_pct={NEAR_PCT:.0%} | hold={HOLD_DAYS}d or resistance"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    # volume panel: bar per trading day (same x positions as the price panel
    # above), birth/buy days highlighted and labeled with their ratio to the
    # trailing 10-day average (dashed line)
    vols = [s["volume"] for s in vol_stats]
    avg_vols = [s["avg_volume_10d"] for s in vol_stats]

    def _bar_color(s):
        if s["is_buy_day"] and s["is_birth_day"]:
            return "#9467bd"
        if s["is_buy_day"]:
            return "#d62728"
        if s["is_birth_day"]:
            return "#2ca02c"
        return "#9ecae1"

    bar_colors = [_bar_color(s) for s in vol_stats]
    vax.bar(x, vols, color=bar_colors, width=0.8, zorder=3)
    vax.plot(x, avg_vols, color="black", linestyle="--", linewidth=1, zorder=4)

    for xi, s in zip(x, vol_stats):
        if (s["is_buy_day"] or s["is_birth_day"]) and not np.isnan(s["volume_ratio"]):
            vax.annotate(
                f"{s['volume_ratio']:.2f}x",
                (xi, s["volume"]),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=8, fontweight="bold", zorder=5,
            )

    vax.set_ylabel("volume")
    legend_handles = [
        mpatches.Patch(color="#9ecae1", label="volume"),
        mpatches.Patch(color="#2ca02c", label="birth day"),
        mpatches.Patch(color="#d62728", label="buy day"),
        mlines.Line2D([], [], color="black", linestyle="--", label="trailing 10d avg"),
    ]
    if any(s["is_buy_day"] and s["is_birth_day"] for s in vol_stats):
        legend_handles.append(mpatches.Patch(color="#9467bd", label="birth + buy (same day)"))
    vax.legend(handles=legend_handles, loc="best", fontsize=7)
    vax.grid(alpha=0.25)

    vax.set_xticks(x)
    vax.set_xticklabels(labels, rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_event_rows(ticker, close, i, signal_row, vol_stats, event_id, chart_file):
    """Long-format rows for every day in this event's plotted window -- the
    exact same [lo, hi] slice plot_buy() draws, and the exact same vol_stats
    it charts -- so the CSV always matches what's on the chart. One row per
    (event, date); event-level fields (bands, birth, age) are repeated on
    every row so a row is self-contained without a join.
    """
    buy_date = close.index[i]

    rows = []
    for s in vol_stats:
        rows.append({
            "event_id": event_id,
            "ticker": ticker,
            "chart_file": chart_file,
            "buy_date": buy_date.date(),
            "date": s["date"],
            "day_offset": s["j"] - i,
            "close": float(close.iloc[s["j"]]),
            "is_buy_day": s["is_buy_day"],
            "is_birth_day": s["is_birth_day"],
            "volume": s["volume"],
            "avg_volume_10d": s["avg_volume_10d"],
            "volume_ratio": s["volume_ratio"],
            "support_low": signal_row["support_low"],
            "support_high": signal_row["support_high"],
            "support_birth": signal_row["support_birth"],
            "support_age_days": signal_row["support_age_days"],
            "resist_low": signal_row["resist_low"],
            "resist_high": signal_row["resist_high"],
        })
    return rows


def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = random.choice(config.ticker_list)
    print(f"picked ticker: {ticker}")

    combo = load_fixed_combo()
    print(f"pulling full history for {ticker}...")
    close, volume = pull_series(ticker)
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    if volume is not None:
        volume = volume.copy()
        volume.index = pd.to_datetime(volume.index)
    print(f"{len(close)} trading days, {close.index[0].date()} -> {close.index[-1].date()}")

    print("running walk-forward simulation (levels recomputed from scratch every day)...")
    buys = run_walkforward(ticker, close, combo)
    print(f"{len(buys)} buy signal(s) found")

    if not buys:
        print("no buy signals for this stock over its history under the fixed combo/rule -- try another ticker")
        return

    out_dir = os.path.join(CHART_DIR, ticker)
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for i, signal_row in buys:
        buy_date = close.index[i].date()
        event_id = f"{ticker}_{buy_date}"
        chart_file = f"{event_id}.png"
        out_path = os.path.join(out_dir, chart_file)
        vol_stats = window_volume_stats(close, volume, i, signal_row)
        plot_buy(ticker, close, i, signal_row, vol_stats, out_path)
        all_rows.extend(build_event_rows(ticker, close, i, signal_row, vol_stats, event_id, chart_file))
        print(f"  buy {buy_date} @ {signal_row['price']:.2f} -> {out_path}")

    events_df = pd.DataFrame(all_rows, columns=EVENT_COLUMNS)
    events_csv_path = os.path.join(out_dir, f"{ticker}{EVENTS_CSV_SUFFIX}")
    events_df.to_csv(events_csv_path, index=False)

    print(f"\nsaved {len(buys)} chart(s) to {out_dir}")
    print(f"saved underlying window data ({len(all_rows)} rows, {len(buys)} events) to {events_csv_path}")

    print_volume_summary(events_df)


def print_volume_summary(events_df):
    """Descriptive-only look at whether birth-day / buy-day volume runs hot
    relative to the same window's other days. Paired within event (each
    event compared against its own "other days" mean) so this isn't
    confounded by the stock's volume trending or by cross-event scale
    differences -- but it's still one stock's ~20 events, so this is a feel
    check, not a claim.
    """
    other = events_df[~events_df.is_buy_day & ~events_df.is_birth_day]
    other_by_event = other.groupby("event_id")["volume_ratio"].mean()

    def _describe(label, mask):
        sub = events_df[mask].dropna(subset=["volume_ratio"])
        if sub.empty:
            print(f"{label}: no data")
            return
        ratios = sub["volume_ratio"]
        paired = (sub.set_index("event_id")["volume_ratio"] - other_by_event).dropna()
        t, p = stats.ttest_1samp(paired, 0.0) if len(paired) > 2 else (float("nan"), float("nan"))
        print(f"{label} (n={len(ratios)}): mean ratio={ratios.mean():.2f}x  median={ratios.median():.2f}x  "
              f"| vs. same-window other days: mean diff={paired.mean():+.2f}x, t={t:.2f}, p={p:.3f}")

    print("\n=== volume vs. trailing 10-day average, birth/buy days vs. rest of window ===")
    _describe("birth day  ", events_df.is_birth_day)
    _describe("buy day    ", events_df.is_buy_day)
    print(f"other days   (n={len(other)}): mean ratio={other['volume_ratio'].mean():.2f}x  "
          f"median={other['volume_ratio'].median():.2f}x  (baseline, by construction ~1x)")


if __name__ == "__main__":
    main()
