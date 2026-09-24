"""Visualize a sample of two-touch-low buy signals (two_touch_low_daybyday.py,
locked params from mean_reversion_notes.md 2026-09-22), price +/-5 trading
days around the buy signal, with the low (L), rebound (L+1), retest (L+2,
rebound_retest branch only), buy day and exit day marked, and the rule's
parameters printed on the chart.

Reads two_touch_low_daybyday_trades.csv (already produced by
two_touch_low_daybyday.py) rather than re-running the signal, and samples a
few winners/losers from each branch so the charts are representative, not
cherry-picked toward one outcome.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_charts.py [n_per_bucket]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from tech_level_live import pull_all
from two_touch_low_daybyday import PARAMS, HOLD, VOL_LOOKBACK, volume_ratio

TRADES_CSV = os.path.join(HERE, "two_touch_low_daybyday_trades.csv")
CHART_DIR = os.path.join(HERE, "data", "two_touch_low_charts")
WINDOW = 5  # trading days shown before/after the buy signal


def sample_trades(df, n_per_bucket, seed=0):
    df = df.copy()
    df["win"] = df.ret_net > 0
    picked = []
    for branch in df.branch.unique():
        for win in (True, False):
            bucket = df[(df.branch == branch) & (df.win == win)]
            if len(bucket):
                picked.append(bucket.sample(min(n_per_bucket, len(bucket)), random_state=seed))
    return pd.concat(picked) if picked else df.iloc[0:0]


def window_bounds(n, i):
    return max(0, i - WINDOW), min(n - 1, i + WINDOW)


def plot_signal(ticker, close, volume, row, out_path):
    idx = close.index
    L = idx.get_loc(pd.Timestamp(row.low_date))
    buy_i = idx.get_loc(pd.Timestamp(row.entry_date))
    exit_i = idx.get_loc(pd.Timestamp(row.exit_date))
    L1 = L + 1
    # buy_i IS L + retest_gap for the rebound_retest branch (whatever gap
    # was used to generate this trade), so this stays correct regardless
    # of which retest_gap the trades CSV was produced with.
    L2 = buy_i if row.branch == "rebound_retest" else None

    n = len(close)
    lo, hi = window_bounds(n, buy_i)
    x = list(range(lo, hi + 1))
    labels = [idx[j].strftime("%Y-%m-%d") for j in x]

    fig, (ax, vax) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})

    ax.plot(x, close.iloc[lo:hi + 1].values, color="#1f5fa8", linewidth=1.6,
             marker="o", markersize=3, label=ticker)

    def mark(pos, color, marker, label, size=110):
        if lo <= pos <= hi:
            ax.scatter([pos], [close.iloc[pos]], color=color, s=size, zorder=5,
                       marker=marker, label=label)

    gap = buy_i - L  # retest_gap for this trade (2 for flat's own L+1, or whatever gap generated it)
    mark(L, "#2ca02c", "v", f"L (first low) {idx[L].date()}")
    mark(L1, "#ff7f0e", "^", f"L+1 (rebound) {idx[L1].date()}")
    if L2 is not None:
        mark(L2, "#9467bd", "s", f"L+{gap} (retest) {idx[L2].date()}")
    mark(buy_i, "#d62728", "*", f"buy {idx[buy_i].date()} @ {close.iloc[buy_i]:.2f}", size=200)
    mark(exit_i, "black", "x", f"exit {idx[exit_i].date()} @ {close.iloc[exit_i]:.2f}", size=140)

    ax.axhline(close.iloc[L], color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_ylabel("close")
    gap_note = f", retest_gap={gap}" if row.branch == "rebound_retest" else ""
    ax.set_title(
        f"{ticker} -- {row.branch} buy {idx[buy_i].date()}{gap_note}  "
        f"(pct1={row.pct1:+.2%}, ret_net={row.ret_net:+.2%}, excess={row.excess_ret:+.2%})"
    )
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(alpha=0.25)

    param_text = "locked params:\n" + "\n".join(f"  {k} = {v}" for k, v in PARAMS.items())
    ax.text(0.01, 0.02, param_text, transform=ax.transAxes, fontsize=7.5,
            family="monospace", va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999999"))

    # volume panel with volume_ratio annotated at L / L+1 / L+2
    vols = [float(volume.iloc[j]) if not np.isnan(volume.iloc[j]) else 0.0 for j in x]
    avg_vols = [None] * len(x)
    for k, j in enumerate(x):
        seg = volume.iloc[max(0, j - VOL_LOOKBACK):j]
        avg_vols[k] = float(seg.mean()) if len(seg) and seg.notna().any() else np.nan

    def bar_color(j):
        if j == buy_i:
            return "#d62728"
        if j == L:
            return "#2ca02c"
        if j == L1:
            return "#ff7f0e"
        if L2 is not None and j == L2:
            return "#9467bd"
        return "#9ecae1"

    vax.bar(x, vols, color=[bar_color(j) for j in x], width=0.8, zorder=3)
    vax.plot(x, avg_vols, color="black", linestyle="--", linewidth=1, zorder=4)

    marked = {L: "L", L1: "L+1"}
    if L2 is not None:
        marked[L2] = f"L+{gap}"
    for j in x:
        if j in marked:
            vr = volume_ratio(volume, j)
            if not np.isnan(vr):
                vax.annotate(f"{marked[j]}: {vr:.2f}x", (j, volume.iloc[j]),
                             textcoords="offset points", xytext=(0, 4), ha="center",
                             fontsize=7.5, fontweight="bold", zorder=5)

    vax.set_ylabel("volume")
    legend_handles = [
        mpatches.Patch(color="#9ecae1", label="volume"),
        mpatches.Patch(color="#2ca02c", label="L"),
        mpatches.Patch(color="#ff7f0e", label="L+1"),
        mlines.Line2D([], [], color="black", linestyle="--", label=f"trailing {VOL_LOOKBACK}d avg"),
    ]
    if L2 is not None:
        legend_handles.insert(3, mpatches.Patch(color="#9467bd", label=f"L+{gap}"))
    vax.legend(handles=legend_handles, loc="best", fontsize=7)
    vax.grid(alpha=0.25)

    vax.set_xticks(x)
    vax.set_xticklabels(labels, rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pending(ticker, close, volume, L, diag, out_path):
    """Same style as plot_signal(), for a candidate that hasn't resolved
    yet -- no buy/exit to mark, window centered on the most recent bar
    ("today") instead of a buy day. diag is a two_touch_low_daybyday.
    diagnose() result, used for the title only."""
    idx = close.index
    n = len(close)
    today_i = n - 1
    L1 = L + 1

    lo, hi = window_bounds(n, today_i)
    x = list(range(lo, hi + 1))
    labels = [idx[j].strftime("%Y-%m-%d") for j in x]

    fig, (ax, vax) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})

    ax.plot(x, close.iloc[lo:hi + 1].values, color="#1f5fa8", linewidth=1.6,
             marker="o", markersize=3, label=ticker)

    def mark(pos, color, marker, label, size=110):
        if lo <= pos <= hi:
            ax.scatter([pos], [close.iloc[pos]], color=color, s=size, zorder=5,
                       marker=marker, label=label)

    mark(L, "#2ca02c", "v", f"L (first low) {idx[L].date()}")
    if L1 <= today_i:
        mark(L1, "#ff7f0e", "^", f"L+1 (rebound) {idx[L1].date()}")
    mark(today_i, "#17becf", "o", f"today {idx[today_i].date()} @ {close.iloc[today_i]:.2f}", size=150)

    ax.axhline(close.iloc[L], color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6)

    ax.set_ylabel("close")
    stage = diag["stage"]
    bits = [f"{k}={v:+.2%}" if k == "pct1" else f"{k}={v:.2f}x"
            for k, v in diag.items() if k in ("pct1", "vr_L", "vr_L1")]
    ax.set_title(f"{ticker} -- PENDING ({stage})  {', '.join(bits)}")
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(alpha=0.25)

    param_text = "locked params:\n" + "\n".join(f"  {k} = {v}" for k, v in PARAMS.items())
    ax.text(0.01, 0.02, param_text, transform=ax.transAxes, fontsize=7.5,
            family="monospace", va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#999999"))

    vols = [float(volume.iloc[j]) if not np.isnan(volume.iloc[j]) else 0.0 for j in x]
    avg_vols = [None] * len(x)
    for k, j in enumerate(x):
        seg = volume.iloc[max(0, j - VOL_LOOKBACK):j]
        avg_vols[k] = float(seg.mean()) if len(seg) and seg.notna().any() else np.nan

    def bar_color(j):
        if j == L:
            return "#2ca02c"
        if j == L1:
            return "#ff7f0e"
        if j == today_i:
            return "#17becf"
        return "#9ecae1"

    vax.bar(x, vols, color=[bar_color(j) for j in x], width=0.8, zorder=3)
    vax.plot(x, avg_vols, color="black", linestyle="--", linewidth=1, zorder=4)

    marked = {L: "L"}
    if L1 <= today_i:
        marked[L1] = "L+1"
    for j in x:
        if j in marked:
            vr = volume_ratio(volume, j)
            if not np.isnan(vr):
                vax.annotate(f"{marked[j]}: {vr:.2f}x", (j, volume.iloc[j]),
                             textcoords="offset points", xytext=(0, 4), ha="center",
                             fontsize=7.5, fontweight="bold", zorder=5)

    vax.set_ylabel("volume")
    legend_handles = [
        mpatches.Patch(color="#9ecae1", label="volume"),
        mpatches.Patch(color="#2ca02c", label="L"),
        mpatches.Patch(color="#ff7f0e", label="L+1"),
        mpatches.Patch(color="#17becf", label="today"),
        mlines.Line2D([], [], color="black", linestyle="--", label=f"trailing {VOL_LOOKBACK}d avg"),
    ]
    vax.legend(handles=legend_handles, loc="best", fontsize=7)
    vax.grid(alpha=0.25)

    vax.set_xticks(x)
    vax.set_xticklabels(labels, rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    n_per_bucket = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    gap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    suffix = "" if gap in (None, 2) else f"_gap{gap}"
    trades_csv = os.path.join(HERE, f"two_touch_low_daybyday_trades{suffix}.csv")
    if not os.path.exists(trades_csv):
        print(f"{trades_csv} not found -- run "
              f"two_touch_low_daybyday.py 21 20{f' {gap}' if gap else ''} first")
        return
    df = pd.read_csv(trades_csv, parse_dates=["low_date", "entry_date", "exit_date"])
    if gap is not None:
        # gap only affects the rebound_retest branch -- flat is identical to
        # the locked (gap=2) run, so restrict to the branch that changed.
        df = df[df.branch == "rebound_retest"]
    sample = sample_trades(df, n_per_bucket)
    print(f"sampled {len(sample)} trades: \n{sample.groupby(['branch', sample.ret_net > 0]).size()}\n")

    tickers = sorted(sample.stock.unique())
    series, failed = pull_all(tickers)
    if failed:
        print(f"no data for: {failed}")

    os.makedirs(CHART_DIR, exist_ok=True)
    close_cache, volume_cache = {}, {}
    for t in tickers:
        c, v = series[t]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        close_cache[t] = c
        if v is not None:
            v = v.copy()
            v.index = pd.to_datetime(v.index)
            volume_cache[t] = v.reindex(c.index)
        else:
            volume_cache[t] = pd.Series(np.nan, index=c.index)

    for _, row in sample.iterrows():
        t = row.stock
        out_path = os.path.join(CHART_DIR, f"{t}_{row.entry_date.date()}_{row.branch}{suffix}.png")
        plot_signal(t, close_cache[t], volume_cache[t], row, out_path)
        print(f"  {t} {row.entry_date.date()} ({row.branch}, ret_net={row.ret_net:+.2%}) -> {out_path}")

    print(f"\nsaved {len(sample)} chart(s) to {CHART_DIR}")


if __name__ == "__main__":
    main()
