"""Visualize two-touch-low v2 buy signals (two_touch_low_v2_daybyday.py)
at the sweep's combined-best combo (two_touch_low_v2_sweep.py, same
seed=21/k=20 sample, n_back widened to include 3-4), price +/-5 trading
days around the buy signal, with the first trough (L), second trough /
buy day (L2) and exit marked, an intraday high-low whisker at L and L2 so
the buyers-took-over condition can be checked by eye, the close-tolerance
band the retest had to land in, and the rule's parameters printed on the
chart.

Re-runs the signal (there's no saved trades CSV for the combined-best
combo -- the sweep only ever kept summary stats) rather than reading one,
using the exact same run_ticker/decide() as the daybyday script, so
nothing here can drift from what was actually swept.

Not a cherry-picked sample: takes each ticker's most recent fired signal
in the seed=21/k=20 universe and plots up to n_charts of them.

Run (from repo root):
  ./venv/bin/python strategies/five_day_bounce/experiments/two_touch_low_v2_charts.py [n_charts] [seed] [k]
"""
import datetime
import os
import random
import sys
from multiprocessing import Pool, cpu_count

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
from tech_level_naive_strategy import equal_weight_curve, attach_benchmark
from tech_level_live import pull_all
from tech_level_continuation_live import LIVE_TICKERS
from two_touch_low_v2_daybyday import run_ticker, is_local_min, buyers_took_over, volume_ratio, HOLD

VOL_LOOKBACK = 10  # matches volume_ratio()'s default trailing window

# combined-best combo from two_touch_low_v2_sweep.py (widened n_back grid,
# seed=21/k=20 sample), 2026-09-23 -- not yet cross-validated on a
# different sample, see mean_reversion_notes.md caveats before trusting it
BEST_PARAMS = dict(
    n_back=5,
    n_fwd=3,
    volume_threshold_L=1.0,
    volume_threshold_L2=0.8,
    close_tolerance=0.005,
)
DD_PCT = 0.01
WINDOW = 5  # trading days shown before/after the buy signal
CHART_DIR = os.path.join(HERE, "data", "two_touch_low_v2_charts")


def window_bounds(n, i):
    return max(0, i - WINDOW), min(n - 1, i + WINDOW)


def gate_label(bto, vr, threshold):
    if bto and not (not np.isnan(vr) and vr > threshold):
        return "BTO"
    if not bto and not np.isnan(vr) and vr > threshold:
        return f"vol {vr:.2f}x"
    if bto:
        return f"BTO + vol {vr:.2f}x"
    return "?"


def plot_signal(ticker, close, high, low, volume, row, p, out_path):
    idx = close.index
    L = idx.get_loc(pd.Timestamp(row.low_date))
    buy_i = idx.get_loc(pd.Timestamp(row.entry_date))
    exit_i = idx.get_loc(pd.Timestamp(row.exit_date))
    L2 = buy_i

    n = len(close)
    lo, hi = window_bounds(n, buy_i)
    x = list(range(lo, hi + 1))
    labels = [idx[j].strftime("%Y-%m-%d") for j in x]

    fig, (ax, vax) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})

    ax.plot(x, close.iloc[lo:hi + 1].values, color="#1f5fa8", linewidth=1.6,
             marker="o", markersize=3, label=ticker, zorder=3)

    # intraday high-low whisker at L and L2, close marked with a tick --
    # this is what buyers_took_over is actually testing, shown so it can
    # be checked by eye rather than taken on faith
    for j in (L, L2):
        if lo <= j <= hi and not (pd.isna(high.iloc[j]) or pd.isna(low.iloc[j])):
            ax.plot([j, j], [low.iloc[j], high.iloc[j]], color="#666666",
                     linewidth=2.5, alpha=0.6, zorder=2)
            ax.plot([j - 0.15, j + 0.15], [close.iloc[j], close.iloc[j]],
                     color="#666666", linewidth=2.5, alpha=0.9, zorder=2)

    # close-tolerance band the retest had to land inside, anchored on close[L]
    tol = p["close_tolerance"]
    c_L = float(close.iloc[L])
    band_lo, band_hi = c_L * (1 - tol), c_L * (1 + tol)
    ax.axhspan(band_lo, band_hi, xmin=0, xmax=1, color="#2ca02c", alpha=0.08, zorder=1)
    ax.axhline(c_L, color="#2ca02c", linestyle="--", linewidth=0.9, alpha=0.6)

    def mark(pos, color, marker, label, size=110):
        if lo <= pos <= hi:
            ax.scatter([pos], [close.iloc[pos]], color=color, s=size, zorder=5,
                       marker=marker, label=label)

    vr_L = volume_ratio(volume, L)
    vr_L2 = volume_ratio(volume, L2)
    bto_L = buyers_took_over(high, low, close, L)
    bto_L2 = buyers_took_over(high, low, close, L2)

    mark(L, "#2ca02c", "v", f"L (first trough) {idx[L].date()} [{gate_label(bto_L, vr_L, p['volume_threshold_L'])}]")
    mark(L2, "#d62728", "*", f"L2 (buy) {idx[L2].date()} @ {close.iloc[L2]:.2f} "
                              f"[{gate_label(bto_L2, vr_L2, p['volume_threshold_L2'])}]", size=200)
    mark(exit_i, "black", "x", f"exit {idx[exit_i].date()} @ {close.iloc[exit_i]:.2f} ({row.exit_reason})", size=140)

    ax.set_ylabel("close")
    ax.set_title(
        f"{ticker} -- two-touch-low v2 buy {idx[L2].date()}  "
        f"(ret_net={row.ret_net:+.2%}, excess={row.excess_ret:+.2%})"
    )
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(alpha=0.25)

    param_text = "params (combined-best, sweep 2026-09-23):\n" + "\n".join(
        f"  {k} = {v}" for k, v in p.items()) + f"\n  dd_pct = {DD_PCT}  hold = {HOLD}"
    ax.text(0.01, 0.02, param_text, transform=ax.transAxes, fontsize=7.5,
            family="monospace", va="bottom", ha="left", zorder=6,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95, edgecolor="#999999"))

    vols = [float(volume.iloc[j]) if not np.isnan(volume.iloc[j]) else 0.0 for j in x]
    avg_vols = [None] * len(x)
    for k, j in enumerate(x):
        seg = volume.iloc[max(0, j - VOL_LOOKBACK):j]
        avg_vols[k] = float(seg.mean()) if len(seg) and seg.notna().any() else np.nan

    def bar_color(j):
        if j == L2:
            return "#d62728"
        if j == L:
            return "#2ca02c"
        return "#9ecae1"

    vax.bar(x, vols, color=[bar_color(j) for j in x], width=0.8, zorder=3)
    vax.plot(x, avg_vols, color="black", linestyle="--", linewidth=1, zorder=4)

    marked = {L: "L", L2: "L2"}
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
        mpatches.Patch(color="#d62728", label="L2 (buy)"),
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
    n_charts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    tickers = random.Random(seed).sample(LIVE_TICKERS, k)
    print(f"seed {seed}: {', '.join(tickers)}", flush=True)
    print(f"params: {BEST_PARAMS}, dd_pct={DD_PCT}\n", flush=True)

    series, failed = pull_all(tickers + ["SPY"], include_hl=True)
    if failed:
        print(f"no data for: {failed}")
    today = datetime.date.today()
    close, high, low, volume = {}, {}, {}, {}
    for t in tickers:
        if t not in series:
            continue
        c, h, l, v = series[t]
        c = c.copy(); c.index = pd.to_datetime(c.index)
        mask = c.index.date < today
        close[t] = c[mask]
        h = h.copy() if h is not None else pd.Series(np.nan, index=c.index)
        h.index = pd.to_datetime(h.index)
        high[t] = h.reindex(close[t].index)
        l = l.copy() if l is not None else pd.Series(np.nan, index=c.index)
        l.index = pd.to_datetime(l.index)
        low[t] = l.reindex(close[t].index)
        if v is not None:
            v = v.copy(); v.index = pd.to_datetime(v.index)
            volume[t] = v.reindex(close[t].index)
        else:
            volume[t] = pd.Series(np.nan, index=close[t].index)
    ew = equal_weight_curve(close)
    spy_close_raw = series["SPY"][0].copy()
    spy_close_raw.index = pd.to_datetime(spy_close_raw.index)
    spy_close = {t: spy_close_raw.reindex(close[t].index, method="ffill") for t in close}

    args = [(t, close[t], high[t], low[t], volume[t], spy_close[t], BEST_PARAMS, DD_PCT) for t in close]
    with Pool(min(len(close), cpu_count())) as pool:
        out = pool.map(run_ticker, args)
    df = pd.DataFrame([row for o in out for row in o])
    if df.empty:
        print("no trades fired at this combo -- nothing to chart")
        return
    df = attach_benchmark(df, ew)
    print(f"{len(df)} total signals across {df.stock.nunique()} tickers\n")

    # one chart per ticker (most recent signal), up to n_charts tickers
    latest = df.sort_values("entry_date").groupby("stock").tail(1)
    if len(latest) > n_charts:
        latest = latest.sample(n_charts, random_state=seed)
    else:
        print(f"only {len(latest)} distinct ticker(s) fired at this combo in this sample "
              f"(requested {n_charts})")

    os.makedirs(CHART_DIR, exist_ok=True)
    for _, row in latest.sort_values("stock").iterrows():
        t = row.stock
        out_path = os.path.join(CHART_DIR, f"{t}_{row.entry_date.date()}_v2.png")
        plot_signal(t, close[t], high[t], low[t], volume[t], row, BEST_PARAMS, out_path)
        print(f"  {t} {row.entry_date.date()} (ret_net={row.ret_net:+.2%}) -> {out_path}")

    print(f"\nsaved {len(latest)} chart(s) to {CHART_DIR}")


if __name__ == "__main__":
    main()
