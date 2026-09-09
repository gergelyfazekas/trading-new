# Technical-level parameter validation — session notes

## Motivation

Replace the hand-tuned technical (support/resistance) level detector — whose
parameters (`tech_width`, `find_peaks` sensitivity, ...) were chosen by eyeballing
charts via `tuning.tune_tech_levels`'s manual `g`/`b` labeling loop — with a
quantitative, walk-forward validation: build levels from price history, check
whether price actually turns (hit) or slides through (miss) when it revisits a
level, and score parameter combos on that basis, per stock rather than forcing
one global winner (a combo that's excellent for a handful of sideways-moving
stocks and mediocre elsewhere is a legitimate, tradeable outcome, not noise to
filter out).

## Live status

**As of 2026-08-06, AAPL and AMZN are live** for hand-trading — see
`tech_level_live.py` / `data/live_combos.json`. Run this once a day:

```bash
cd /Users/gergelyfazekas/trading-new
./venv/bin/python tech_level_live.py
```

It re-pulls each ticker's full price history via `StockList.pull_data`, rebuilds
levels, and prints the closest support below and resistance above current price
(band bounds, birth date, % distance), plus whether price is currently inside a
band. Read-only decision support — nothing executes, this is for trading by hand.

**Two arms are reported per ticker** (changed 2026-08-06, see "Extending the
live list" below):

- **`fixed`** — one combo, identical for every ticker, chosen *a priori* from
  the universe calibration in `DEFAULT_GRID`: `distance=10, prominence=0.01`
  (universe p50), `tech_width=0.008` (middle of the tight 0.003–0.012 range that
  is structurally immune to the width-inflation artifact). Nothing about it was
  selected on any individual stock's outcome, so **the forward record it
  generates carries no selection bias**. This is the arm to trade off.
- **`tuned`** — that ticker's own grid-search winner (AAPL `distance=5,
  prominence=0.03, tech_width=0.012`; AMZN `distance=5, prominence=0.05,
  tech_width=0.02`). Shown to reveal whether per-name tuning would have mattered,
  *not* to be picked from opportunistically. The script prints an explicit
  agreement line so a disagreement is visible rather than rationalisable.

Only the level **below** is a long-only actionable setup — price falling into a
band from above is a support test, the direction every EV result below was
measured on. Resistance is context only.

To add a ticker: `tech_level_live.save_combo(ticker, combo)` sets its tuned arm;
`set_fixed_combo(combo)` replaces the shared fixed arm (which **resets the
forward record** — events before and after are no longer the same experiment).
`load_combos` still accepts the pre-2026-08 flat `{ticker: combo}` layout.

### The forward record (`data/live_log.csv`)

Every run appends one row per (trading day, ticker, arm) — price, whether it's
inside a band, the support/resistance neighbours with birth dates and %
distances, level count, and **the combo params used**, so a later
`set_fixed_combo`/`save_combo` is visible in the log and can be segmented on
before anything is pooled.

**This file must never be regenerated from history.** Its whole value is that
each row is what the detector actually showed *on that date*, and the newest
~`distance` days of levels are provisional — a local extremum can't be confirmed
until price moves away from it (see the prominence-scaling section). Recomputing
an old date yields a level set the decision could not have used, silently
converting an out-of-sample record into an in-sample one. Consequently it is
also the one file here that can't be rebuilt if lost, so `data/` is no longer
ignored wholesale: `.gitignore` now excludes `data/*` but re-includes
`live_log.csv` and `live_combos.json`. Commit them.

Re-running on the same day upserts on `(as_of, ticker, arm)` rather than
appending duplicates, so a second run after an intraday data revision is safe,
and running on a non-trading day just rewrites the last trading day's rows.

### Scheduling (2026-08-07)

Automated via a **launchd agent**, not cron:
`~/Library/LaunchAgents/com.gergelyfazekas.techlevellive.plist` → `run_live_log.sh`,
**08:00 local, Mon-Fri**.

- **Why 08:00 local.** After the previous US close (22:00 CEST) and long before
  the next open (15:30 CEST), so yfinance's latest bar is always a *settled*
  session. This is not cosmetic: running inside US market hours logs a partial,
  still-forming bar (observed directly — three runs on 2026-08-06 gave 312.06 →
  312.16 → 312.41 for the same `as_of`), and because the upsert is last-write-
  wins it can overwrite an `inside=True` at-level event with a later
  `inside=False` reading. That would systematically erase exactly the short
  excursions the record exists to count. The schedule makes both impossible.
  08:00 stays safe across US/EU DST mismatch weeks (US open lands at 14:30-16:30
  local in every case).
- **Why launchd rather than cron.** cron silently skips a run if the machine was
  asleep; launchd runs a missed occurrence on wake. A skipped weekday is a
  trading day permanently missing from the record, since the script only ever
  sees the *latest* bar.
- **Limit of that guarantee:** launchd recovers a *delayed* run (asleep at 08:00,
  awake at 10:00), not a genuinely absent one. If the machine is off for three
  days, launchd coalesces the missed occurrences into a single run that sees only
  the latest bar — those trading days are still lost. Bounded backfill (dates
  older than ~`distance` trading days have settled level sets, per the truncation
  test's 99.6% agreement) is the only real fix and is not built.
- Retries 5× at 120s: launchd may fire on wake seconds before the network is up.
- Logs to `~/Library/Logs/tech_level_live{,.err}.log` (~400KB/year, appended).

```bash
launchctl list | grep techlevellive                      # is it registered
launchctl kickstart gui/$(id -u)/com.gergelyfazekas.techlevellive   # run now
launchctl bootout gui/$(id -u)/com.gergelyfazekas.techlevellive     # disable
```

At ~10-20 at-level events per name per year across two names, expect a usable
sample around 2028. The point of the exercise is that the `fixed`-arm rows will
need no discount for combo selection — unlike every backtest number above.

## Design decisions

- **Hit = price turns (bounces) at the level. Miss/penalty = price passes
  straight through.**
- **Level construction must be causal.** A level's band freezes at its 2nd
  confirming touch. The original `Stock.get_tech_levels` (stock_class.py)
  clusters touches over a whole window at once, so a much-later touch can widen
  a level's band retroactively — a real lookahead bug, not just a style choice.
  The new construction (`tech_levels.build_levels_causal`) walks touches in
  date order so a level's existence and boundaries only ever depend on touches
  at or before its own birth.
- **Hit/miss is decided directly off the close series** (band entry/exit
  crossings), not off the discrete peak list — otherwise the test would be
  circularly dependent on `find_peaks` sensitivity, which is itself one of the
  tuned parameters.
- **Log everything first, filter later.** Every level/touch is recorded
  unfiltered; a minimum-touches-per-level filter exists (`min_touches` in
  `tech_level_scoring.score`) but defaults to off.
- **Four scores per `(stock, combo)`**: local / global(leave-one-out) ×
  in-sample / out-of-sample. Global is leave-one-out specifically so a stock's
  own good result can't inflate its own corroboration.
- **A level born after the OOS cutoff is excluded from both IS and OOS
  scoring** — it isn't a fair "did the known signal hold" test either way.
- **Grid search = outer × inner loop.** Outer: parameters that change
  `find_peaks` itself (`prominence`, `distance`, `height`, `rel_height`,
  `consider_volume`, `volume_height`, `volume_prominence`) — reruns detection.
  Inner: `tech_width` — only re-clusters already-found touches, cheap.
  (`tech_level_search.run_stock_grid`)
- **Per-stock winners are the goal, not a single global param set.** A combo
  that's great for 2 stocks and useless elsewhere is a legitimate result to act
  on (trade those stocks specifically), not something to average away.
- Old strong/medium touch-count buckets (`get_tech_levels`'s 0/0.5/1 strength)
  are retired conceptually in favor of the continuous empirical hit rate.
- Visual inspection via **Streamlit** (`tech_level_app.py`), not notebook/
  matplotlib-only, for fast side-by-side comparison of a good- vs. bad-scoring
  combo on the same stock.

## Files

- `tech_levels.py` — core mechanics: `find_touches`, `build_levels_causal`,
  `score_touches`, `run_combo_for_stock`, `levels_to_rows`, `combo_key`, `Level`.
  Dependency-light, no I/O.
- `tech_level_search.py` — grid definition (`DEFAULT_GRID`) + outer/inner-loop
  driver, parallelized across stocks with `ProcessPoolExecutor`, writes one CSV
  per stock into `config.tech_level_folder` (overwrites per stock on each run).
- `tech_level_scoring.py` — `load_log` + `score` (the four metrics, plus
  `n_touches_is`/`n_touches_oos`/`n_levels`/`n_touches` diagnostics).
- `tech_level_app.py` — Streamlit inspector: pick a stock, cutoff date,
  min-touches filter, and two combos (defaulting to best/worst local OOS); see
  both charts side by side with bands anchored at birth date and hit/miss
  markers, plus the four scores.
- `config.py` — added `tech_level_folder`.

## Verification approach

- Hand-built synthetic bounce / passthrough / still-open-at-series-end price
  series to check `score_touches` classifies all three correctly.
- Hand-computed leave-one-out arithmetic for a small crafted log and matched it
  exactly against `tech_level_scoring.score`.
- Ran the real `ProcessPoolExecutor` grid search against synthetic stocks to
  confirm the multiprocessing driver behaves.
- Used `streamlit.testing.v1.AppTest` to run the actual app headlessly (first
  against fake data, later against real data) and confirm no exceptions, correct
  combo-picker defaults, and that both charts render.

## Real end-to-end run (first time populating real data)

- Set up a project venv (`trading-new/venv`) + `.gitignore` (`venv/`,
  `__pycache__/`, `.DS_Store`).
- Created the data directory tree under
  `~/Documents/python_projects/trading_new_structure/data/` (didn't exist yet).
- Pulled bronze via `yfinance` for all 40 tickers in `config.ticker_list`
  (2015-05-28 → present), computed silver (returns/variance/SMA/RSI) and gold
  (SMA crosses) exactly as `pull_data.ipynb` does — deliberately skipping the
  old `tech_level_input_calc` step, since that's what this whole system replaces.

## Grid-tuning findings

1. Calibrated `prominence`/`distance`/`tech_width` ranges against real
   peak-prominence distributions (sampled AAPL/KO/JNJ/XOM/MMM/DIS/T/BA) instead
   of guessing. More volatile names need proportionally higher prominence at
   any given percentile (BA ~3x AAPL's prominence at p50-p95).
2. **`rel_height` was inert** in the original grid — `scipy.find_peaks` only
   uses it alongside a `width` constraint, which was never passed. Removed it.
3. **Found and fixed a real gap in `tech_level_scoring.score()`**: it exposed
   only whole-history `n_levels`/`n_touches` diagnostics, not the actual
   in-sample/out-of-sample touch counts behind `local_is`/`local_oos`. This let
   spurious "perfect" 100% results through, driven by 2-3-touch OOS samples.
   Added `n_touches_is`/`n_touches_oos` — filter on these, not `n_touches`.
4. Correlation between `local_is` and `local_oos`, pooled across the whole
   grid × universe, is weak (~0.09) — most of this large-cap universe isn't
   fertile ground for this idea, as expected.
5. Tested "sideways stocks should score better" directly with a crude
   trend-ratio (net drift ÷ cumulative vol budget). Correlation with mean OOS
   hit rate was weak (~0.20) and didn't cleanly separate winners from losers
   (AAPL/GOOG/GS — all high-trend — scored well; NKE — genuinely flat — scored
   among the worst). Full-history trendiness is too coarse a filter on its own.
6. **Major finding: widening `tech_width` from 0.003 to 0.05 makes `local_oos`
   *and* `global_loo_oos` climb together, monotonically, across the board**
   (~49% → ~60%) — the "wide bands trivially win" failure mode anticipated
   during design, showing up for real. Raw `local_oos` alone was a misleading
   ranking metric; it conflates genuine local edge with generic width inflation
   every stock gets.
7. Corrected the ranking to **excess = `local_oos` − `global_loo_oos`**
   (the width-matched baseline), and required that excess to be positive as the
   *minimum* across four independent cutoffs (2021-06, 2022-06, 2023-01,
   2024-01), not just at one convenient date. AAPL/HD — the apparent best
   results under the flawed raw-`local_oos` ranking — did not survive this.
   Standouts that did:
   - **KO**: distance=5, prominence=0.015, tech_width=0.03 — worst-case excess
     +11.4%, mean +15.8%
   - **MMM**: distance=10, prominence=0.015, tech_width=0.003 *and* 0.005 (same
     stock, two neighboring widths both holding up — a good robustness sign)
   - **DHR**: distance=10, prominence=0.01, tech_width=0.012 — +9.6%
   - **MO**: distance=5, prominence=0.01, tech_width=0.003 — +7.9%

## Volume: first pass (`consider_volume=True`)

Added `tech_level_search.expand_outer_with_flag` — a helper for grid params
gated behind a boolean flag (`volume_height`/`volume_prominence` are inert
whenever `consider_volume=False`, so blindly crossing them into `expand_outer`
just produces duplicate, differently-labeled combos for every False case).
`run_search`'s `grid['outer']` now also accepts a pre-built combo list, not
just a spec dict, so this can feed straight in.

Calibrated `volume_prominence`/`volume_height` the same way as the price-side
params, against real volume z-score peak-prominence distributions (p75-p90
sits roughly 0.5-2.5 across this universe). Ran a combined grid: 12 price-only
(`consider_volume=False`) combos + 48 volume-enabled combos, × 7 `tech_width`
values, × 40 stocks (~9M log rows, 2.8GB).

**Result: mixed, and one important correction.** In aggregate, `consider_volume`
barely moves the excess-over-baseline distribution either way. The
volume-enabled combos that looked like the biggest wins (KO in particular,
worst-case excess up to +14.6% vs. its price-only +11.4%) were concentrated at
`tech_width=0.03-0.05` — exactly the width range already flagged as prone to
the "wide bands trivially win" artifact. Checked directly: **KO's own excess
climbs monotonically with `tech_width` (near 0 at 0.008-0.02, up to 0.35 at
0.05) with or without volume** — the leave-one-out global correction doesn't
fully net this out when a stock's own susceptibility to the width artifact
differs from the pooled average, so KO's apparent "win" is substantially
artifact, not signal. Downgraded confidence in it accordingly.

DHR and MMM, by contrast, show the *opposite* shape — excess is highest at the
**tightest** widths (0.003-0.012) and decays toward the wide end — which is
structurally immune to the width-inflation mechanism, so these remain credible.
For both, adding volume touches didn't clearly help at their best (tight-width)
setting (MMM: 0.124 → 0.006 excess with volume on; DHR: 0.153 → 0.089) — if
anything, volume-based touch candidates diluted the effect slightly, likely by
pulling in extra, less-relevant candidate touches at a scale that blurs an
otherwise clean tight level.

**Takeaway: `consider_volume=True` as a touch-discovery mechanism (find
high-volume days, treat them as extra candidate touches) is not a clear win in
this data** — worth remembering before spending more grid-search time on it.
This doesn't rule out other ways of using volume (see conversation) — using
volume as *context* on an existing touch (e.g. was a passthrough on unusually
high volume) is a different mechanism entirely and hasn't been tried yet.

## Volume: second pass (context on existing touches + forward-return check)

Built the two pieces from the conversation's "volume as context" idea (not
touch discovery, which was tested above and was weak):

- **`tech_levels.score_touches(level, close, volume=None)`** now also records
  a volume z-score (same whole-history normalization `find_touches` uses) on
  each touch's *resolving* date (the day the excursion actually exits — the
  breakout day for a miss, the bounce-back day for a hit). `Level.touches`
  entries are now `(date, outcome, volume_zscore)` triples; `levels_to_rows`
  adds a `touch_volume_zscore` log column. This is populated for *every*
  combo regardless of `consider_volume` (that flag only affects touch
  discovery, not this). One subtlety worth remembering: `touch_date` stays
  entry-anchored (unchanged, matches existing chart/scoring conventions), so
  `touch_volume_zscore` can refer to a later calendar date than `touch_date`
  itself when an excursion takes several days to resolve — they describe the
  same event but aren't necessarily the same day.
- **`tech_level_continuation.py`** (new module, separate from
  `tech_level_scoring.py` since it needs raw price series, not just the log):
  `add_forward_returns(log, stock_list, n_days)` computes, for every miss, the
  n-day-forward return in the breakout direction (positive = continuation,
  negative = reversion), and `compare_by_volume` splits misses into high/low
  `touch_volume_zscore` (pooled median split) and compares.

**Result: the hypothesis (high-volume passthrough → continuation, low-volume
→ reversion) doesn't show a clear signal.** Pooled across the whole universe
(n≈267k misses per bucket), both high- and low-volume buckets show *negative*
mean forward return at every horizon tested (5/10/20 days) — i.e. on average,
price mean-reverts somewhat after breaking a level regardless of volume — and
the gap between buckets is small and doesn't hold direction consistently
across horizons (high-volume less negative at 5/10 days, more negative at 20).
Restricting to just the DHR/MMM credible standouts (~244 misses total) doesn't
help either — direction flips at the 5-day horizon and the bucket means at
10/20 days are within roughly one standard error of each other given the small
sample. Not ruling the idea out, but there's no evidence for it yet with what's
been tried (whole-history z-score, pooled median split, these three horizons).

## Prominence scaling bug — found and fixed 2026-07-21

Found while building `tech_level_input.py` (features for the forecasting
pipeline). A truncation test — recompute features on the first 70% of history,
check the shared dates still match — failed on **all 16 feature columns across
most of the history**. Cause was in `find_touches`:

```python
scaled = (close / close.iloc[-1]).to_numpy()   # old
```

Dividing by the final price made `prominence` an **absolute currency threshold**
rather than a fractional one. For AAPL that was `0.005 × 333.74 = $1.67` — which
is 5.7% of the 2015 price but 0.5% of the 2026 price. Two consequences:

1. **Non-uniform detection across history.** The detector was ~11x stricter at
   the start than at the end: 4-7 touches/year in 2015-17 vs 26-31/year in
   2021-25. That gradient runs along the *same axis as the IS/OOS split*, so
   in-sample and out-of-sample were never measured with the same detector.
2. **Lookahead.** Truncating the series moved the threshold, so the same date
   gained or lost touches depending on where history happened to end (129 vs
   191 touches over an identical date range). Anything derived from it was
   non-causal.

**Fix:** `find_peaks` now runs on `log(close)`, so prominence is a fractional
move at any price level, scale-free and dependent on no series-wide constant.
Truncation test in log space: 99.6% touch agreement, and the feature check now
matches on every shared date except the last 4 before the cut — a detection-lag
boundary effect (a local extremum can't be confirmed until price moves away from
it), not a leak. Practical consequence: **the newest ~`distance` days of level
features are provisional and can revise as data arrives.**

Log space also compressed the cross-sectional spread of prominence (p50 runs
0.0066 for KO to 0.0167 for AMAT, ~2.5x, vs the ~3x under the old scaling), so
`DEFAULT_GRID`'s prominence range was recalibrated to `[0.01, 0.02, 0.03, 0.05]`
— spanning universe p50 to p90, roughly 27 touches/year down to 11.

### What changed in the results, and what didn't

Full regrid (84 combos × 40 stocks, 1.22M log rows), rescored on the same
4-cutoff worst-case-excess criterion with a ≥20 OOS touches per cutoff filter:

- **Finding #6 (width inflation) is real and independent of this bug.** Mean
  `local_oos` still climbs 48.6% → 60.9% as `tech_width` goes 0.003 → 0.05.
  The excess metric still nets it out correctly (excess stays within ±0.011
  across the whole width range).
- **DHR and MMM survive again, still at the tightest widths** (0.003-0.005) —
  the same shape as before, structurally immune to width inflation. The two
  credible standouts survived a change that reshuffled everything upstream of
  them, which is the strongest evidence yet that they're real.
- **KO is still flagged suspicious by the width-shape check** — its excess
  still climbs monotonically with width (−0.224 at 0.003 to +0.158 at 0.05).
  Same artifact, reproduced independently. The diagnostic is stable.
- **AAPL has zero combos surviving the strict bar** (positive excess at all 4
  cutoffs *and* ≥20 OOS touches at each), consistent with it failing the
  4-cutoff check before. Re-picked anyway for the live script, with the sample
  caveat stated explicitly — see "Re-picking AAPL" below.
- New names clearing the bar: DIS (width curve *decays*, slope −0.023, 60+ OOS
  touches — credible on shape), T, NKE, ACN.

**Multiple-comparisons caveat, now much sharper.** 349 of 1482 sufficiently
sampled (stock, combo) pairs survive, spread over 29 of 40 stocks. With 84
combos tried per stock, at least one survivor per stock is close to expected by
chance. **"Best combo per stock" is a heavily selection-biased statistic** —
treat the width-shape check and cross-fix persistence (DHR/MMM) as the real
evidence, not the ranking position.

### Re-picking AAPL (2026-07-21)

AAPL's `≥20 OOS touches at every cutoff` filter and its actual signal point in
opposite directions: the filter only admits `tech_width ≥ 0.02`, but AAPL's
excess is *highest* at the tightest widths and decays as width grows (mean
excess +0.31 at 0.005 → +0.02 at 0.05; slope vs log(width) −0.112). So the
sufficiently-sampled combos are exactly the ones where the signal has already
faded. The strict bar returns nothing.

Ranking tight-width combos by raw worst-case excess walks straight into finding
#3's trap — the top four are 100% hit rates on 1-5 total OOS touches. Ignored.

Picked instead by *family coherence*: `prominence=0.03, distance=5` is positive
at all four cutoffs across four consecutive widths (0.003/0.005/0.008/0.012) and
goes negative at 0.02+. Within that family `tech_width=0.012` is the best
compromise — it has both the highest worst-case excess (+0.148) and the most
data (99 OOS touches, min 9 per cutoff, mean hit rate 69.3%):

| tech_width | worst excess | n_pos/4 | mean hit | total OOS touches |
|---|---|---|---|---|
| 0.003 | +0.005 | 4 | 0.819 | 24 |
| 0.005 | +0.155 | 4 | 0.869 | 31 |
| 0.008 | +0.089 | 4 | 0.695 | 75 |
| **0.012** | **+0.148** | **4** | **0.693** | **99** |
| 0.020 | −0.023 | 3 | 0.581 | 182 |

**Caveat to keep in mind while hand-trading it: this does not meet the ≥20
touches-per-cutoff bar that DHR/MMM clear.** min 9 at the weakest cutoff. It's
the best-supported AAPL combo available, not a validated one — weaker evidence
than the DHR/MMM standouts, and it should not be promoted to anything automated
on this basis.

## Wiring into the forecasting panel (2026-07-21)

`tech_level_input.py` turns levels into per-date features; `forecast.build_panel`
assembles the long panel with them attached.

- **`level_features`** emits 16 `tl_*` columns per (date, stock): distance to the
  nearest level below/above, whether price is inside a band, and per-level touch
  count / shrunk hit rate / age / band width / days-since-touch. Touch counts and
  hit rates are keyed on the *resolution* date, not `touch_date` — an outcome
  isn't knowable until the excursion exits the band.
- **`select_combos(log, selection_cutoff)`** picks each stock's combo and a
  confidence weight from **in-sample excess only** (`local_is - global_loo_is` at
  the cutoff). Deliberately not the 4-cutoff OOS excess: that's computed over the
  whole history, so selecting on it and training on earlier dates would leak the
  future into the panel. `build_panel` truncates the panel to start strictly
  after `selection_cutoff` to keep this honest.
- Selection lands mostly on **tight widths** (13 stocks at 0.003, 8 at 0.005),
  and `corr(confidence, tech_width) = -0.32` — the in-sample selector is *not*
  chasing the width-inflation artifact, which was the main worry.
- `build_pipeline` now caps `max_features=0.5` (was sklearn's 1.0 default) so
  dominant features can't crowd the level columns out of every split.
- `NON_FEATURE_COLS` drops the gold tier's duplicated date columns
  (`dt`, `dt.1`-`dt.3`) and redundant `ticker`; leaving them in makes the panel
  non-numeric and `RandomForest.fit` raises.

### Results — and how much of the first number was the wrong target

First measured on raw next-day `log_return`, the level features looked strong:
mean daily cross-sectional IC +0.044 vs a +0.004 baseline, positive in all four
walk-forward folds (a single 75/25 split flattered it further still, +0.062 with
t=5.6 and an +18bp/day top-8 spread).

**Most of that did not survive moving to the correct target.** On the vol-scaled
10-day target, with 10-day purging between train and test (targets now overlap,
so adjacent folds share information without it):

| fold start | baseline (no `tl_`) | with `tl_` |
|---|---|---|
| 2022-06 | −0.023 | +0.018 |
| 2023-06 | −0.023 | −0.013 |
| 2024-06 | +0.001 | +0.069 |
| 2025-06 | +0.038 | +0.009 |
| **mean IC** | **−0.002** | **+0.021** |
| **mean top-8 spread** | **−0.024** | **+0.043** (sd units) |

So the honest number is roughly **+0.02, positive in 3 of 4 folds** — about half
the original, and no longer consistent in sign. The original figure was inflated
by two things the vol-scaled target removes: at a 1-day horizon the `tl_`
distances pick up short-horizon mean reversion that has largely decayed by 10
days, and on *unscaled* returns cross-sectional IC partly rewards ranking stocks
by volatility, since high-vol names simply move more.

What does hold up: `tl_` beats the baseline in every fold, and mean top-8 spread
flips from negative to positive. +0.02 IC is a plausible magnitude for a real
signal, where +0.06 was not.

**Also found while re-testing:** the gold tier's raw price and volume *levels*
(`close`/`high`/`low`/`open`/`sma_9`-`sma_100`/`volume`) were actively harmful.
Trees split on absolute values, so in a pooled panel these act as stock-identity
and regime labels that fit the training window and generalize to nothing — the
baseline was anti-predictive (mean IC −0.018, negative in all four folds) until
they were dropped. Now in `forecast.NON_STATIONARY_COLS`, excluded by default.

### Time-proxy test — the +0.021 does not hold up

Five `tl_*` columns are monotone counters that could act as calendar proxies:
`tl_n_levels` (+0.29 correlation with time), `tl_below_age` / `tl_above_age`
(+0.08), `tl_below_days_since_touch` / `tl_above_days_since_touch` (+0.02/+0.05).
Tested on the vol-scaled 10-day target:

| arm | mean IC |
|---|---|
| baseline (no `tl_`) | −0.002 |
| with `tl_` (all 17) | +0.021 |
| `tl_` **minus** the 5 proxies | **+0.005** |
| baseline **+ only** the 5 proxies | +0.016 |

Removing them costs three quarters of the effect; they recover most of it on
their own. That alone is ambiguous — these columns vary both over time *and*
across stocks on a given date, and cross-sectional variation would be a
perfectly legitimate signal ("this stock is choppier than its peers today").

The disambiguating test: replace each proxy with its **within-date
cross-sectional rank**, which preserves the ordering across stocks and destroys
the calendar component (verified: correlation with time goes to exactly 0.000).
If the edge were cross-sectional it would survive.

| arm | mean IC |
|---|---|
| with `tl_`, raw proxies | +0.021 |
| with `tl_`, proxies rank-within-date | **+0.006** |

It does not survive. Ranked, the result collapses to +0.006 — essentially
identical to deleting the proxies outright (+0.005). **So the apparent edge was
not cross-sectional information in these features; it was their time-linked
level.** (A third arm, baseline + ranked proxies only, printed +0.020, but
entirely from one fold (+0.083) with the other three at or below zero — noise,
not a counterexample.)

**Conclusion: the level features do not currently demonstrate a reliable
cross-sectional edge on the vol-scaled 10-day target.** The defensible
contribution is ~+0.005 IC over baseline, which at this sample size is not
distinguishable from nothing.

Worth being clear that this does *not* contradict the DHR/MMM hit-rate work.
Those measured "when price reaches a known band, does it bounce or pass
through" — a statement about **path**, at a level's own irregular timing. The
panel model asks "rank 40 stocks by 10-day forward return, every day". A real
bounce tendency need not produce a tradeable cross-sectional return ranking:
it says nothing about magnitude, and its timing doesn't align to a fixed
10-day grid. Both results can be true at once.

**Still open:**
- Fold 2 (2023-06) is negative for both arms. Unexplained.
- The methodology itself (grid, shortlist, universe) was developed with
  full-history knowledge. `select_combos` fixes the per-stock choice; it can't
  fix that.
- If the level signal is to enter the forecast at all, it probably needs to be
  posed the way it was validated — e.g. an event-conditional target ("return
  *given* price is currently at a level"), not a feature in a daily
  every-stock-every-day panel.

## Trade-level P&L — the missing number (2026-07-21)

`tech_level_trades.py`. Everything before this measured *direction only* (does
price bounce), never what a bounce is worth. Trade rule: enter at the close of
band entry, exit `hold_days` after a bounce confirms, stop out immediately at
the close that passes through. Benchmarked against the **equal-weight universe
over the identical window** — from an equal-weight base an overweight only pays
if the name beats the rest of the book, not if it merely rises — net of 10bp.

Only **support** tests are actionable long-only (price falls into the band from
above, expected to bounce up). Resistance tests need a short; the most an
equal-weight book can express is dropping to zero weight, ~−2.5%.

### Three findings

**1. `score_touches`' hit rate conflates support and resistance, and the edge
isn't in the tradeable direction.** DHR — one of the two credible standouts —
has a **48.6% support-only hit rate**, a coin flip. Its validated edge lives
substantially in the resistance direction, which long-only can barely express.
A limitation of the original validation design, not a new failure.

**2. Payoff asymmetry is ~1:1** (gain/loss ratios 0.91–1.07 for DHR/MMM/DIS), so
breakouts don't dwarf bounces as feared. But 1:1 payoffs at 49–59% accuracy net
to roughly zero, and costs push them negative.

**3. Pooled EV is zero at every holding period.** Across the four names
(n≈303 support trades): EV +0.0001 (hold 0), +0.0004 (5), −0.0011 (10), +0.0014
(20); |t| ≤ 0.52 throughout. MMM and DIS get actively *worse* with longer
holding — their average gain on a confirmed bounce turns negative by hold 10 —
which is direct evidence against the bounce-and-ride thesis.

### The AAPL exception, and how far it survives scrutiny

AAPL is the only positive name, and it improves monotonically with holding:
EV +0.38% → +0.87% → +1.12% → +1.56% per trade at hold 0/5/10/20, t≈2.2–2.6,
cumulative excess +137% over 88 trades at hold 20. Its whole advantage is a
62.9% hit rate combined with a 2.04 gain/loss ratio.

Because all of that gain comes from holding winners longer (stop-outs are
unaffected by `hold_days` by construction), the obvious confound is that AAPL
simply beat the equal-weight book. **Placebo test:** 2000 draws of random AAPL
entries, same trade count, same holding-length distribution, same date window.
Random entries earn +0.20% (hold 10) / +0.46% (hold 20) — so drift explains only
a fraction. The actual result sits at the **98th percentile, p≈0.02**.

**Not enough.** AAPL is one name out of 40, with a combo picked from 84; p≈0.02
does not survive that selection exposure. And note the inversion: the names that
*passed* hit-rate validation (DHR/MMM) have negative EV, while the name that
*failed* it has the positive one — so hit rate is not predictive of trade
profitability, exactly as the payoff-asymmetry argument predicted.

**Conclusion: do not build the portfolio overlay on this evidence.** The
structure (equal-weight base, deviate only on level events, revert after) is
sound and correctly matched to how the signal was validated — the data just
doesn't support it. AAPL is already being hand-traded daily; that accumulates
genuinely out-of-sample trades at ~10-20/year and is the cleanest way to settle
it, on a 1-2 year timescale rather than another backtest.

## Extending the live list (2026-08-06)

Question asked: add GOOGL, NVDA, AMZN, MSFT, TSLA to the daily script. Answer
after checking: **only AMZN earned a place.** What the checks found, in order.

### Hit-rate survival is at the base rate for these names

GOOG and AMZN both have combos clearing the strict 4-cutoff bar that AAPL never
managed, which looks like a promotion — until compared against the pooled
survival rate of 23.5% (349/1482 sufficiently-sampled pairs):

| stock | survivors / sampled | rate | binomial p |
|---|---|---|---|
| DIS | 48/84 | 57% | 4e-11 |
| T | 44/81 | 54% | 2e-9 |
| ACN | 27/44 | 61% | 1e-7 |
| GOOG | 6/22 | 27% | **0.42** |
| AMZN | 9/55 | 16% | **0.93** |
| MSFT | 0/1 | — | — |

GOOG sits *on* the base rate; AMZN is *below* it; MSFT has one sufficiently
sampled combo in the entire 84-combo grid. **A raw survivor count is not
evidence** — this is the multiple-comparisons caveat made quantitative, and it
should be applied to any future shortlist.

### Ranking on EV, and the drift confound

Per the earlier finding that hit rate doesn't predict profitability, candidates
were ranked on support-only trade EV net of 10bp vs. the equal-weight book, then
placebo-tested against 2000 random-entry draws per name matched on trade count,
date window, and holding-length distribution (hold=10):

| stock | n | real EV | random EV | lift | p |
|---|---|---|---|---|---|
| AAPL | 89 | +1.118% | +0.204% | +0.914% | **0.018** |
| CSCO | 118 | +0.535% | −0.105% | +0.640% | **0.017** |
| AMZN | 112 | +0.529% | −0.016% | +0.544% | 0.135 |
| GOOG | 147 | +0.375% | +0.224% | +0.151% | 0.333 |
| MSFT | 370 | +0.276% | +0.145% | +0.131% | 0.248 |

**GOOG's raw EV is 60% drift; MSFT's is 53%.** Neither has a real lift. The
mega-caps top the raw EV table only because they beat an equal-weight book of 40
large caps over 2015-2026 no matter when you enter — always placebo-test before
reading a positive EV here. Pooled EV across all 40 names is **−0.091%,
t=−2.11**, unchanged from the earlier conclusion.

At 40 names tested, 4 land under p<0.05 where 2 are expected, and the membership
reshuffles by horizon (hold 10: AAPL/COST/AXP/CSCO; hold 20: AAPL/AMZN/CSCO).
That instability is itself evidence most of these are noise.

### The width sweep — which reversed the CSCO/AMZN ordering

CSCO's selected combo sits at `tech_width=0.05`, the range flagged as untrust-
worthy. That warning was about hit rate and CSCO's claim is an EV claim, so EV
was swept across width directly, placebo-testing each (lift %, hold=20):

| width | 0.005 | 0.008 | 0.012 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|---|
| **CSCO** | −0.21 | −0.29 | −0.13 | +0.04 | +0.51 | **+1.06** (p=.002) |
| **AMZN** | +3.23 | +2.30 | +1.20 | +1.16 | +1.16 | +1.13 |
| **AAPL** | −0.15 | +0.56 | **+1.08** (p=.028) | +0.64 | **+0.80** (p=.009) | −0.37 |

- **CSCO is the KO artifact reproduced in EV space** — negative at every width
  from 0.005 to 0.012 and positive only at the widest band. Its p=0.017/0.003
  came entirely from `tech_width=0.05`. **Dropped, despite having the second-best
  headline p-value.** The width-shape check remains the single most useful
  diagnostic in this whole project.
- **AMZN has the most robust shape of any name tested** — positive lift at
  *every* width, decaying gently (structurally immune to width inflation),
  against a random-entry baseline of ~0. Its single-combo p=0.135 understated it;
  the width sweep is the stronger evidence. **Added.**
- **AAPL is peaked, not monotone** — significant at 0.012 and 0.03, negative at
  0.003/0.005 and 0.05. Not the artifact shape, but width-sensitive enough that
  the live combo sits on a local optimum. **Kept, with that caveat.**

### Decisions

- **Live list: AAPL + AMZN.** GOOGL and MSFT have no lift over drift; CSCO is a
  width artifact.
- **NVDA and TSLA were not added.** Both are in `config.oos_ticker_list`, the
  reserved validation universe — grid-searching them to pick a combo would burn
  them. (Applying a *pre-chosen* combo and trading forward would not, so the
  fixed arm is a legitimate way to include them later if wanted.)
- **The fixed arm exists because per-name combo selection is the known weakness.**
  Every result above had to be discounted for selection over 84 combos; a combo
  fixed a priori and applied uniformly generates a forward record that needs no
  such discount. That is the number worth waiting for.
- The a-priori fixed combo's own backtest, for the record: AAPL lift +1.167%
  (p=0.022) and AMZN +0.739% (p=0.059) at hold 20; CSCO negative, again. Chosen
  from grid calibration before these were computed, not after.

## Naive support-strategy backtest, a fresh OOS split, and the level-age finding (2026-09-07)

Question: backtest the literal hand-trading rule (buy within `near_pct` of a
support, sell at `hold_days` or on touching resistance, whichever first) —
`tech_level_naive_strategy.py`, entry/exit mechanics documented in its own
docstring. Two things came out of doing this properly.

### A second, genuinely untouched split: `config.oos_ticker_list`

`tech_level_naive_strategy.py` ran on `config.ticker_list` (the 40 names used
throughout this file) and does not select any parameter from the outcome — it
uses the fixed a-priori combo only — but `ticker_list` itself is the universe
`DEFAULT_GRID` was calibrated against, so a cleaner test runs the same fixed
combo against `config.oos_ticker_list`'s 60 disjoint names instead.
`tech_level_oos_strategy.py` does this, loading the frozen `data/oos/*.csv`
pull (same convention as `vol_regime.py`/`volume_model.py`/`alpha.py`) rather
than a fresh yfinance call.

**Found a contamination this reservation was supposed to prevent:**
`data/tech_levels/NVDA.csv` already holds a full 84-combo grid-search touch
log (mtime 2026-07-20), even though this file says (above, 2026-08-06) NVDA
was deliberately *not* grid-searched because it's reserved. The log predates
that note, so someone ran it before the reservation was written down. It
doesn't bias the fixed combo (that combo's params came from `ticker_list`
calibration, not NVDA's grid results), but NVDA is no longer an uninspected
name, so `tech_level_oos_strategy.py` excludes it — 59 of the 60 names used.
**Flag for any future OOS use of this list: verify a name has no file under
`data/tech_levels/` before trusting it as untouched.**

First run, unmodified `tech_levels.py` mechanics: pooled hold=5, near_pct=1%
excess over equal-weight was **-0.03% (t=-1.31)** — flat, consistent with (and
independent confirmation of) this file's running suspicion that the
`ticker_list` results ride on drift/multiple-comparisons rather than a real
effect.

### Level invalidation (`tech_levels.mark_broken`)

Plotting AVGO/ORLY price with every level ever born (`build_levels_causal`
bands are permanent, by design, so the grid-search hit/miss scoring can see
what happens after a break) made the actual problem visible: a stock that's
grown 10-20x drags along a decade of stale bands nobody would call "support"
today, and — the important part — a level that price has already broken
straight through keeps being fed to the entry rule as if it still meant
something, in whichever direction (support/resistance) it now happens to sit.

Added `tech_levels.mark_broken(levels, close)`: causally scans each level's
post-birth close series and sets `Level.broken_at` to the first date price is
seen fully on the opposite side of the band from where it was first resolved
(a true pass-through, not just a touch). Deliberately **not** wired into
`build_levels_causal`/`score_touches` — the grid-search/scoring pipeline
(`tech_level_search.py`, `tech_level_scoring.py`, `tech_level_continuation.py`)
needs post-break touches to compute hit rates and is untouched.
`tech_level_naive_strategy.build_levels()` calls it; `simulate()` excludes any
level once `date >= broken_at`. `tech_level_live.py`'s daily script was left
on the old (permanent-level) definition — not yet decided whether it should
change too.

Re-running with this filter: pooled hold=5 excess jumped to **+0.41%
(t=8.18)** on the OOS universe (3,835 trades, down from 18,210) and **+0.27%
(t=4.20)** on `ticker_list` independently. Passed the standard checks before
trusting it: positive in **12/12 calendar years** on OOS, **48/59** stocks
individually positive, trimmed mean (+0.34%) and median (+0.31%) close to the
raw mean (not outlier-driven), and it's a structural rule applied uniformly
rather than a parameter swept against the outcome, so it doesn't reopen the
selection-bias problem the fixed combo exists to avoid.

### Level age at entry — the edge is not what it looks like

Sensitivity sweep (`tech_level_sensitivity.py`) checked three things:
`near_pct`/`hold_days` grid robustness (positive across near_pct 0.5-3% and
hold_days 1-10 on both universes — not a lucky cell), cost sensitivity
(breakeven ~40-50bp round trip on OOS, thinner at ~35-40bp on `ticker_list`,
comfortable margin either way over the assumed 10bp), and — the one that
matters — **how old the support level was when the trade entered**:

| age since birth | OOS mean excess (t) | ticker_list mean excess (t) |
|---|---|---|
| 0-5 days | **+0.98% (t=13.3)** | **+1.07% (t=10.6)** |
| 5-20 days | +0.02% (t=0.19) | -0.24% (t=-2.01) |
| 20-60 days | +0.02% (t=0.18) | -0.35% (t=-2.63) |
| 60-180 days | -0.15% (t=-1.07) | -0.23% (t=-1.09) |
| 180+ days | +0.18 to +0.34% (t<1) | -0.88% to -0.01% (mostly negative) |

**The entire pooled edge lives in trades entered within 5 days of the level's
birth.** Everything older is statistically zero on OOS and actually
*negative* and significant on `ticker_list` for the 5-365 day buckets — the
opposite of "support holds up over time."

Read together with `mark_broken`'s effect: filtering to "still unbroken"
disproportionately keeps *young* levels, because given enough history almost
every old level eventually gets crossed while a 2-day-old level hasn't had
the chance yet. So the jump from -0.03% to +0.41% was never evidence that
technical support/resistance is real — it was `mark_broken` inadvertently
enriching the sample toward freshly-formed levels, and freshly-formed levels
are where a short-lived post-swing-low continuation effect lives. **The
correct description of what's been found is: price that just carved out a
two-touch swing low tends to keep drifting in that direction for about 5
trading days — a known-shape short-horizon continuation/reversal pattern, not
"price respects a known level."** Mature, previously-validated levels show no
edge (OOS) or a real negative one (`ticker_list`).

Not yet done: test the 0-5-day continuation hypothesis directly (entries
gated on level age rather than break status) against its own placebo,
including whether it survives once you exclude names/periods already flagged
suspicious above (KO, width artifacts) and whether it's already fully
explained by ordinary short-horizon price reversal unconditional on any level
existing at all.

## Going live on the continuation effect (2026-09-07)

Built `tech_level_continuation_live.py` to trade the short-horizon
continuation effect from the section above for real feedback: buy within 1%
above a support that is unbroken and <=5 calendar days old, sell at 5 trading
days held or on touching the resistance active at entry. Fixed a-priori
combo only, frozen exactly as validated -- no re-tuning before going live,
which would reopen the selection-bias question the fixed arm exists to
avoid. `config.ticker_list` (40 names), matching how the effect was
validated. `simulate()` (backtest) and the live script now share one
`active_support_resistance()` helper (`tech_level_naive_strategy.py`) so the
two can't silently drift apart.

Two things checked before shipping, both because trading a *young*-level
effect is a different, harder problem than trading an old one:

1. **Detection lag.** `find_peaks` needs subsequent price action to confirm a
   touch, so a live script can't know about a touch on its true date --
   only once enough later bars exist to satisfy `distance`/`prominence`. The
   backtest's `build_levels_causal` was run once over the *whole* history, so
   it implicitly assumes a touch is known exactly on its true date, which a
   live script can't replicate. Measured directly (AVGO, last 3 years, 94
   touches): median 1 trading day lag, mean 1.33, max 4, all 94 detected
   within a week. Small enough not to worry about, but real -- a live
   "5-day-old level" signal is shifted about a day later than the same
   backtest label.
2. **Provisional levels -- the more serious one.** Every level young enough
   to qualify for this rule (<=5 calendar days) is *younger than
   `distance`=10 sessions*, which is exactly the zone `tech_level_live.py`'s
   `provisional_mask` already found real: an AAPL band appeared for 5
   sessions then was un-born once more price data arrived (n_levels
   105->104). The backtest's touches, computed once over the complete
   history, only ever include touches that turned out to be real in
   hindsight -- it never saw the false starts a live reader also has to sit
   through. **This means the +0.41%/+0.27% (t=8.18/4.20) backtest numbers
   above are likely optimistic**, in a way no amount of re-slicing the same
   precomputed data can fix. Going live is the actual fix, not a stopgap: a
   day-by-day script can only ever act on what's currently detectable, so its
   forward record is automatically free of this hindsight bias. Every buy
   row logs `provisional` (age in sessions <= distance) — will essentially
   always be True by construction, logged to check later rather than assumed.

`tech_level_live.py` (AAPL/AMZN/fixed-arm daily report, `data/live_log.csv`
since 2026-08-06) is retired as the daily target but **not deleted** --
its download/retry/settle-check plumbing (`pull_all`, `bar_is_settled`,
`_sessions_since`) is imported directly into the new script rather than
duplicated, and its own forward record stays as historical data, not
extended further. `data/live_log.csv` and `data/live_combos.json` are
still untracked in git as of this writing -- **commit them before anything
touches that directory**, since they can't be rebuilt if lost.

New files, none of them backtest artifacts -- all real forward state:
`data/continuation_positions.json` (mutable, current open positions),
`data/continuation_signal_log.csv` (append-only daily decisions, same
never-recompute-the-past rule as `live_log.csv`), `data/continuation_trades.csv`
(append-only closed-trade ledger -- the actual number to judge this against
once it accumulates).

**Full write-up published as "The Five-Day Bounce"** (Claude artifact,
2026-09-07): premise, mechanics, the flat-then-fixed-then-reframed backtest
arc, the code chain, and the current live setup, in one reviewable page --
https://claude.ai/code/artifact/d1a24516-6991-49e7-937f-15b75feee622
(private; use the page's share menu to hand it to anyone else). Rebuild it
from this file if the link ever breaks -- everything in it traces back to a
section above.

## Execution timing decision (2026-09-08)

**Decision: trade near the actual close, same session, rather than waiting
for the next-morning automated report.** `tech_level_continuation_live.py`
runs once daily at 08:00 CEST, after the prior US close settles -- so
strictly waiting for it means acting a full trading session later than the
close the signal was actually computed from. That is a bigger mismatch
against the backtest (which assumes entry precisely at the triggering close)
than a few minutes of same-day slippage, so buying near the close you can
already see forming is closer to the validated behavior, not further from
it.

What's actually at risk doing this: the entry distance test and
`mark_broken`'s break check both depend on *today's settled close* --
nothing else (a level's band, birth date, and age are already fixed by prior
days and can't change intraday). So the last few minutes can only flip two
things: price drifting outside the 1% zone, or price dipping into/through
the band and marking that level broken as of today. Both are the exact
outcomes this approach is disproportionately exposed to, precisely because
it trades right at a threshold -- but for a liquid large cap absent a
scheduled catalyst, closing-minute moves big enough to matter are rare.

Conditions for doing this: only when price sits comfortably inside the zone
(~0.3-0.7% away, not sitting right at the 0.9-1.0% edge), and not on a day
with a scheduled catalyst (earnings, macro/Fed releases, known
rebalance/opex dates) where closing-minute moves stop being small even for
megacaps.

## Does closing inside the support band also count as a buy? (2026-09-08)

Question raised while reading `tech_level_continuation_live.py`'s entry check
against its own docstring: `active_support_resistance()` (`tech_level_naive_strategy.py`)
only treats a level as "support" when the close sits *strictly above* the
band (`band[1] < price`). A close that lands **inside** the band
(`band_low <= close <= band_high`) fails that check entirely -- it isn't
support and isn't resistance either, so the live script logs it as `watch`
with no buy, even though "price is sitting right at the level" reads, on
its face, like the strongest version of "near support," not a non-event.

Tested directly rather than assumed: reran the exact live rule (fixed
combo, `near_pct=1%`, `hold_days=5`, support age <=5 calendar days --
matching the 2026-09-07 sections above) on both universes, with a modified
`active_support_resistance` where a level containing the close also
qualifies as support (`dist` clamped to 0). Baseline reproduced the
already-published age<=5d numbers (OOS +0.94%, t=13.0 vs the documented
+0.98%, t=13.3 -- close enough to trust the replication) before trusting the
modified version.

| | OOS, age<=5d | ticker_list, age<=5d |
|---|---|---|
| current rule (close strictly above band) | **+0.94%** (t=13.0, n=1640) | **+1.00%** (t=10.5, n=1148) |
| + close-inside-band also buys | **-0.11%** (t=-2.0, n=5415) | **-0.12%** (t=-1.9, n=3788) |
| ...the new (inside-band) trades alone | -0.14% (t=-2.5, n=5271) | -0.15% (t=-2.3, n=3683) |
| ...the original above-band trades that still fired | +0.89% (t=3.4, n=144) | +0.93% (t=4.1, n=105) |

**Verdict: don't extend the rule -- it kills the edge, on both universes.**
Two compounding effects: (1) the inside-band trades are themselves negative,
not just weaker -- a close already inside the band is a different setup
(already testing/violating) from one approaching and holding above it,
consistent with this file's read of the effect as a rejection-continuation,
not generic level-proximity; (2) with one position per ticker at a time,
inside-band entries fire far more often than the narrow 1%-above zone and
crowd out the good trades' slots -- only 144/1640 (OOS) and 105/1148
(`ticker_list`) of the original entries still get to fire once they compete
for the same slot. `active_support_resistance`'s strictly-above-only
definition is left unchanged; `tech_level_watchlist.py`'s BUY BAND display
was showing `[support_low, support_high*1.01]`, which visually implied the
whole band buys -- corrected to `[support_high, support_high*1.01]` to match
what the live script actually trades.

## A next-day stop-loss on the continuation entry (2026-09-08)

Question: after a buy signal, is it worth holding to `hold_days`=5 (or the
resistance touch) if the *very next* close already breaks back through the
level -- or should that be cut immediately instead? The current rule (live
and backtested) has no downside protection at all: it only ever exits on
`hold_days` or a resistance touch, never on the trade going wrong.

Tested as a same-entries overlay, not a new backtest: identical entries to
the validated live rule (fixed combo, near_pct=1%, hold_days=5, support age
<=5 calendar days), only the exit changed -- if the close on the day right
after entry breaks back through a threshold, exit right there instead of
following the normal resistance/hold_days exit. Two thresholds tested, since
"breaks the level" is ambiguous: `stop_low` = next close < `band_low` (fully
back through the whole support box); `stop_high` = next close < `band_high`
(gives back the buy-zone edge, may still be inside the band).

| | OOS, age<=5d (n=1640) | ticker_list, age<=5d (n=1149) |
|---|---|---|
| baseline (no stop) | +0.938% (t=13.0) | +1.002% (t=10.5) |
| + stop_low | **+0.987% (t=14.3)**, 38/1640 (2.3%) tripped | **+1.041% (t=11.2)**, 29/1149 (2.5%) tripped |
| + stop_high | +0.950% (t=13.9), 108/1640 (6.6%) tripped | +1.019% (t=11.1), 74/1149 (6.4%) tripped |

Both look like improvements on excess return, but excess nets out the
benchmark return over each trade's *actual* holding period -- and the stop
trades hold for 1 day instead of ~7, so a different, shorter slice of
benchmark drift gets subtracted. Checked the raw (non-excess) return on just
the tripped trades before trusting either:

| | stop_low tripped (n=38 OOS / 29 ticker_list) | stop_high tripped (n=108 OOS / 74 ticker_list) |
|---|---|---|
| baseline raw ret_net (held anyway) | -4.58% / -4.27% | -0.88% / -1.27% |
| stop raw ret_net (locked at next close) | **-2.22% / -2.66%** | -1.30% / -1.47% |
| baseline hit rate on this subset | 15.8% / 10.3% | 48.1% / 43.2% |

**stop_low is real, stop_high is a benchmark-timing artifact.** For
stop_low, the trades that trip it are already badly broken (15.8%/10.3% hit
rate if held anyway, average loss -4.6%/-4.3%) -- cutting them at the next
close roughly halves the average loss (-2.2%/-2.7%), a genuine raw-return
improvement, not just a shorter benchmark subtraction window (checked: mean
benchmark return over the baseline vs. stop windows was similar for this
subset, -0.9% vs -0.7% OOS). For stop_high, the raw return is *worse* under
the stop than under holding (-1.30% vs -0.88% OOS; -1.47% vs -1.27%
ticker_list) -- the apparent excess improvement there came from the 1-day
stop window happening to catch a down day for the benchmark (mean bench
return -0.31% over the stop's 1-day window vs +0.29% over baseline's ~7-day
window on the OOS tripped subset) — an artifact of the excess metric, not a
real edge from stopping out.

**Recommendation: add stop_low only** -- if the close on the day after entry
is below `support.band[0]`, exit immediately instead of holding to
`hold_days`/resistance. Don't add stop_high. Small sample (38/1640, 2.5% of
trades) so read the exact magnitude cautiously, but the direction and the
raw-return check both hold up. Not yet wired into `tech_level_naive_strategy.simulate()`
or `tech_level_continuation_live.py` -- a live-trading behavior change,
flagged for confirmation before shipping rather than applied on this pass.

## Reorg into strategies/five_day_bounce/, and a watchlist bug (2026-09-09)

**The watchlist bug.** `tech_level_watchlist.py` was silently relabeling any
`watch` row as `BUY` whenever the logged price sat inside BUY BAND, with no
check on `support_age_days` at all. Since `active_support_resistance()` and
the live rule share one `dist` calculation, the *only* way a row logged
`watch` can have price inside BUY BAND is `support_age_days >= MAX_AGE_DAYS`
(dist would already have been in range, so age is the one thing that could
still have blocked a `buy`). That means the override wasn't catching a rare
boundary case as its docstring claimed ("cuts it off by one day") -- it was
unconditionally re-including *every* aged-out level whose price happened to
revisit the zone, which is exactly the 5-20-day-and-older bucket this file's
"Level age at entry" section (2026-09-07, above) found has ~zero excess on
OOS (t=0.19) and negative, significant excess on `ticker_list` (t=-2.01).
Caught 2026-09-09: GILD, 11 days old, shown as BUY. Fixed by making the
watchlist's STATUS column a direct, unmodified read of the log's own `event`
-- it never recomputes or overrides what the live script decided. A
price-in-zone-but-aged-out row still gets surfaced (it's genuinely useful
context -- explains why a familiar-looking level isn't a signal) but now via
a `NOTE` column reading "zone, aged out", never via `STATUS`.

**The reorg.** Moved everything specific to this strategy -- the five scripts
downstream of `tech_levels.py`/`config.py`/`stock_class.py`
(`tech_level_naive_strategy.py`, `tech_level_oos_strategy.py`,
`tech_level_sensitivity.py`, `tech_level_continuation_live.py`,
`tech_level_watchlist.py`), `run_live_log.sh`, and its data
(`continuation_signal_log.csv`, `continuation_positions.json`,
`naive_strategy_trades.csv`, `oos_strategy_trades.csv`) -- into
`strategies/five_day_bounce/`, with a new `NOTES.md` there as the
strategy-specific entry point (what each script does, the code chain, the
frozen rule, how to run things). `tech_levels.py`, `config.py`,
`stock_class.py`, and `tech_level_live.py` stay at the repo root: the first
three are shared by unrelated research scripts (`tech_level_search.py`,
`tech_level_scoring.py`, `tech_level_trades.py`, `tech_level_input.py`,
`tech_level_app.py`) that predate and fed into this strategy but aren't part
of running it, and `tech_level_live.py` is the retired daily script whose
plumbing (`pull_all`, `bar_is_settled`, `_sessions_since`) the live script
here still imports directly.

The frozen combo also got split out: `data/live_combos.json` stays at the
root (still used by `tech_level_live.py` for its per-ticker tuned combos);
`strategies/five_day_bounce/data/fixed_combo.json` is a new file holding just
the frozen `{tech_width: 0.008, distance: 10, prominence: 0.01, ...}` dict,
so this strategy's data folder doesn't depend on the retired script's file.
This file's own dated sections above (paths like `tech_level_watchlist.py`,
`data/naive_strategy_trades.csv`) predate the move and were left as written
-- read them as `strategies/five_day_bounce/<name>`. The launchd agent
(`com.gergelyfazekas.techlevellive.plist`) was repointed at
`strategies/five_day_bounce/run_live_log.sh` and reloaded; `git mv` was used
for the five tracked scripts and `run_live_log.sh`, plain `mv` for the four
untracked data files (all still covered by the same `data/*` gitignore
pattern, now scoped to `strategies/five_day_bounce/data/*` too).

## Open threads / next steps

**Where this stands as of 2026-07-21.** The levels were pursued to feed the
forecasting pipeline (`forecasting_notes.md`). They did not make it in: the
apparent panel IC was a time-proxy artifact, and the trade-level P&L is zero
across the credible names. What remains live is the daily AAPL hand-trading
script, which accumulates genuinely out-of-sample events at ~10-20/year and is
the cleanest way to settle the question — on a 1-2 year timescale, not another
backtest. Re-reading order for picking this up: the prominence-scaling fix, then
the trade-level P&L section, then the time-proxy test.

- **Support and resistance were never separated.** `score_touches` counts
  "bounced up off support" and "rejected down at resistance" identically. DHR's
  support-only hit rate is 48.6% against its pooled ~55% — its edge lives in the
  direction a long-only book can barely express. Any future work on levels
  should split the two from the start; the whole validation history above is
  direction-agnostic and should be read with that in mind.
- **Hit rate turned out not to predict profitability.** The names that passed
  the 4-cutoff validation (DHR/MMM) have negative trade EV; AAPL, which failed
  it, has the only positive one. Direction-only accuracy was the wrong selection
  criterion — any future scoring should rank on EV, not hit rate.
- The volume z-score in `find_touches` / `score_touches` normalizes by
  whole-history `mean()`/`std()` — the same class of leak as the prominence bug,
  untouched because it only affects `consider_volume=True` (already found not to
  help) and the `touch_volume_zscore` annotation.
- Visual inspection of the DHR/MMM (and, with appropriate skepticism, MO/KO)
  shortlist in the app hasn't happened yet.
- Volume-as-touch-discovery and volume-conditioned continuation/reversion have
  both been tried now and neither shows a clear effect in this data — see
  above for exactly what was tried before trying variations (different z-score
  windows, per-stock rather than pooled splits, other horizons, etc.).
- The 4-cutoff consistency check is a practical guard, not a formal
  multiple-comparisons correction — treat any shortlist as promising, not proven.
- Any result concentrated at `tech_width >= 0.03` should be treated with extra
  suspicion regardless of what the leave-one-out excess score says — check the
  per-stock excess-vs-width curve directly before trusting it (see KO above).
  **This now applies to EV results too, not just hit rate** — CSCO passed the
  placebo test at p=0.003 and was still a width artifact (2026-08-06 section).
- **Placebo-test every positive EV before believing it.** Mega-caps beat an
  equal-weight book from any entry point over this window; GOOG's and MSFT's raw
  EV was more than half drift. Random-entry draws matched on count, window and
  holding length are cheap and have overturned two results so far.
- Sweeping a result across `tech_width` and looking at the *shape* has now
  overturned or rescued four names (KO, CSCO, AMZN, AAPL). Prefer it to any
  single-combo p-value — a single combo's significance is a point estimate on a
  curve that selection already picked the top of.
- **`config.oos_ticker_list` has now been spent on this specific question**
  (2026-09-07, on top of `vol_regime.py`/`volume_model.py`/`alpha.py` already
  using it) — reusing the same 60 names to confirm a *different* level-based
  claim later is a weaker check than it looks, since it's no longer a fresh
  cross-section for anything touching technical levels. Also check
  `data/tech_levels/*.csv` for a stray file before trusting any name in this
  list as untouched — NVDA already wasn't (2026-09-07).
- **Before trusting a level/window-based edge, check what it looks like split
  by recency/age of the thing that triggered the entry.** The 2026-09-07
  `mark_broken` result (-0.03% → +0.41% excess) looked like real support/
  resistance behavior and was actually a short-horizon continuation effect
  concentrated entirely in levels <5 days old — filtering on "unbroken"
  inadvertently filtered on "young." A result that survives year/stock/outlier
  checks can still be measuring a completely different mechanism than the one
  it was designed to test.
