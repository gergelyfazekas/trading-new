# The Five-Day Bounce

A short-horizon continuation strategy: buy within 1% above a support level
that was born in the last 5 calendar days, sell at 5 trading days held or on
touching the nearest resistance, whichever comes first. Full writeup with
premise, dead ends, and sensitivity checks:
https://claude.ai/code/artifact/d1a24516-6991-49e7-937f-15b75feee622 (private
artifact) and `tech_levels_notes.md` at the repo root (the source of truth if
that link ever breaks -- every number in the artifact traces back to a dated
section there, particularly "Naive support-strategy backtest, a fresh OOS
split, and the level-age finding (2026-09-07)" and "Going live on the
continuation effect (2026-09-07)"). This file is the map of *this folder*,
not a restatement of the research.

> **WARNING (2026-09-21): the edge described below did not survive a strict
> day-by-day causal test** (levels rebuilt each day from data available that
> day): 318 trades on 20 tickers, hit rate 54% vs 81% in the whole-history
> backtest, excess -0.20% (t=-1.1) vs +0.91%. The earlier "causally confirmed"
> checks only re-tested trades the hindsight backtest had already selected, so
> they missed the trades live trading takes that later get undercut. The rule
> is **not validated**; see `tech_levels_notes.md`, "Day-by-day (fully causal)
> test of the frozen rule" (2026-09-21).

## What it actually is

It is **not** "price respects a known support level." Slicing trades by how
old the triggering level was at entry shows the entire edge lives in levels
younger than 5 days -- everything older is statistically flat (OOS) or
negative (`ticker_list`). The honest description: price that just carved out
a two-touch swing low tends to keep drifting in that direction for about 5
trading days. A level's *band* still matters (it's where price has to be
relative to a real swing low/high), but its *age* is the whole effect.

## The frozen rule (do not re-tune this before trading it)

| parameter | value |
|---|---|
| entry | close strictly above `support.band[1]` (`band_high`), within 1% |
| max level age | < 5 calendar days at entry |
| exit | 5 trading days held, or resistance touched -- first to happen |
| universe | `config.ticker_list` union `config.oos_ticker_list` -- 101 names (config.ticker_list 41 + config.oos_ticker_list 60, disjoint; the published artifact rounds this to "100"), NVDA included |
| combo (find_touches/tech_width) | `distance=10`, `prominence=0.01`, `tech_width=0.008` -- `data/fixed_combo.json` |
| stop-loss | none. A `stop_low` rule (exit if next close < `band[0]`) tested well but is **not wired in** -- see `tech_levels_notes.md`, 2026-09-08 |

Chosen from universe-wide calibration, never from any individual stock's own
outcome, and frozen before going live on 2026-09-07 -- re-tuning it now would
reopen the selection-bias question the fixed arm exists to avoid. If you want
to explore a variant, do it as a new combo/rule tested against a placebo, not
as an edit to the numbers above.

**Execution note (not encoded in any script):** trades happen manually a few
minutes before the US close, buying when price sits comfortably inside the
zone (~0.3-0.7% away, not right at the 0.9-1.0% edge) and skipping days with
a scheduled catalyst (earnings, macro/Fed releases, rebalance/opex). This is
closer to the backtest's same-close entry than waking up to the next
morning's automated report. See `tech_levels_notes.md`, "Execution timing
decision (2026-09-08)".

## The code chain

```
tech_levels.py                    (repo root -- core mechanics, no I/O)
        |  find_touches, build_levels_causal, mark_broken
        v
tech_level_naive_strategy.py      backtest engine: build_levels,
        |                         active_support_resistance (shared with live),
        |                         simulate. Writes data/naive_strategy_trades.csv
        |
        +--> tech_level_oos_strategy.py    same engine, reserved 59-name universe
        |                                  writes data/oos_strategy_trades.csv
        |
        +--> tech_level_sensitivity.py     near_pct x hold grid, cost sweep,
        |                                  age-bucket sweep (prints only)
        |
        +--> tech_level_causal_check.py    causal walk-forward recheck of
        |                                  young-level trades (entry AND
        |                                  exit), non-i.i.d. inference,
        |                                  reversal placebo, concurrency-
        |                                  capped portfolio sim (prints +
        |                                  writes data/causal_check_*.csv,
        |                                  causal_full_*.csv,
        |                                  placebo_reversal_*.csv,
        |                                  portfolio_equity_stats_*.csv)
        |
        v  (frozen rule ships to)
tech_level_continuation_live.py   daily live signal + position tracking.
        |                         Imports pull_all/bar_is_settled/_sessions_since
        |                         from tech_level_live.py (repo root, retired
        |                         script, plumbing only).
        v  writes
data/continuation_positions.json  open positions, overwritten each run
data/continuation_signal_log.csv  append-only, one row per (run_date, ticker)
data/continuation_trades.csv      append-only closed-trade ledger (created on
                                   the first closed trade -- doesn't exist yet)
        |
        v  read by
tech_level_watchlist.py           read-only viewer, no pulling, no computation
```

`config.py`, `stock_class.py`, and `tech_levels.py` stay at the repo root --
they're shared by research scripts that predate this strategy
(`tech_level_search.py`, `tech_level_scoring.py`, `tech_level_trades.py`,
`tech_level_input.py`, `tech_level_app.py`) and aren't needed to run or watch
this one. `tech_level_live.py` also stays at the root: it's the retired
AAPL/AMZN daily report this strategy replaced as the cron target, kept only
because `tech_level_continuation_live.py` still imports its download/retry
plumbing, and because its own forward record (`data/live_log.csv`) is
historical and shouldn't be touched.

## What each file here does

- **`tech_level_naive_strategy.py`** -- the backtest engine. `build_levels()`
  and `active_support_resistance()` are shared verbatim with the live
  script so the two can't silently drift apart. `simulate()` is the backtest
  loop; `run_backtest()` sweeps hold_days, benchmarks against equal-weight,
  and runs a random-entry placebo. Running `main()` re-pulls fresh data for
  `config.ticker_list` and regenerates `data/naive_strategy_trades.csv`.
- **`tech_level_oos_strategy.py`** -- same engine (`run_backtest`, imported)
  on `config.oos_ticker_list` minus NVDA (see its own docstring for why),
  loaded from the frozen `data/oos/*.csv` pull rather than fresh yfinance
  calls, so the OOS confirmation record doesn't drift with when you happen to
  run it. Regenerates `data/oos_strategy_trades.csv`.
- **`tech_level_sensitivity.py`** -- three stress tests on the fixed rule:
  the near_pct x hold_days grid (is 1%/5d a coincidence or does the
  neighborhood hold), a cost_bps sweep (breakeven round-trip cost), and the
  age-bucket sweep that produced the level-age finding above. Prints to
  stdout only, writes nothing.
- **`tech_level_causal_check.py`** -- direct test of the provisional-levels
  caveat, in four steps: (A) rebuild every young-level trade's entry *and*
  exit from data truncated to the relevant date (only data a live trader
  would have had), superseding the hindsight exit; (B) calendar-year block
  bootstrap + ticker-clustered SE, replacing the plain pooled t-stat's i.i.d.
  assumption; (C) a placebo that drops the two-touch level requirement and
  keeps only "near a recently confirmed local low" -- the test of whether
  this is about levels at all; (D) a concurrency-capped portfolio equity
  curve. Plus the original `distance` sensitivity sweep. See
  `tech_levels_notes.md`, 2026-09-07 (original entry-only check) and
  2026-09-15 (this extension). Writes `data/causal_check_ticker_list.csv` /
  `_oos.csv` (entry-only, from the original check), `data/causal_full_*.csv`
  (Step A, entry+exit), `data/placebo_reversal_*.csv` (Step C),
  `data/portfolio_equity_stats_*.csv` (Step D) -- all per-universe,
  regenerable.
- **`tech_level_continuation_live.py`** -- the only script that touches real
  positions. Run once daily, after the prior US close settles: pulls fresh
  closes for all 101 names, evaluates each against open positions or the
  entry rule, and upserts one row per ticker into
  `data/continuation_signal_log.csv` (append-only -- **never regenerate this
  from history**; recomputing an old date with today's fuller history would
  silently convert an out-of-sample decision into an in-sample one). Aborts
  with no writes if the latest bar hasn't settled yet (exit code 3).
- **`tech_level_watchlist.py`** -- read-only viewer of the signal log. Prints
  every one of the 101 names, `BUY`/`HELD`/`SOLD` on top (exactly what the
  live script decided -- this viewer never overrides that), then `watch` rows
  split into "still inside the age window" and "aged out," each sorted by
  distance to the buy zone. A `NOTE` column flags a watch row whose price is
  already inside BUY BAND but whose level has aged out ("zone, aged out") --
  worth seeing, never worth confusing with a signal. See its own module
  docstring for the full column reference and the 2026-09-09 bug this
  replaced (a GILD row shown as BUY at 11 days old -- the old override
  promoted any price-in-zone watch row regardless of age).
- **`run_live_log.sh`** -- retry/timeout wrapper around
  `tech_level_continuation_live.py`, invoked daily at 08:00 local by the
  `com.gergelyfazekas.techlevellive` launchd agent (Mon-Fri). Handles the
  three failure modes that used to lose trading days: transient network
  errors, a hung interpreter, and the machine going to sleep mid-run.

## Data files in `data/`

| file | tracked in git? | mutability |
|---|---|---|
| `fixed_combo.json` | yes | static -- the frozen combo, never changes |
| `continuation_positions.json` | no | freely overwritten every run |
| `continuation_signal_log.csv` | no | append-only forward record -- never regenerate |
| `continuation_trades.csv` | no | append-only closed-trade ledger -- the real P&L record |
| `naive_strategy_trades.csv` | no | regenerable -- rebuilt by `tech_level_naive_strategy.py` |
| `oos_strategy_trades.csv` | no | regenerable -- rebuilt by `tech_level_oos_strategy.py` |
| `causal_check_ticker_list.csv` / `causal_check_oos.csv` | no | regenerable -- rebuilt by `tech_level_causal_check.py` |
| `causal_full_ticker_list.csv` / `causal_full_oos.csv` | no | regenerable -- entry+exit-causal trades, Step A |
| `causal_full_combined.csv` | no | regenerable -- both universes merged, the book actually traded live |
| `placebo_reversal_ticker_list.csv` / `placebo_reversal_oos.csv` | no | regenerable -- Step C placebo trades |
| `portfolio_equity_stats_ticker_list.csv` / `portfolio_equity_stats_oos.csv` | no | regenerable -- Step D summary stats |
| `portfolio_equity_stats_combined.csv` | no | regenerable -- combined-universe portfolio stats quoted in the artifact |

## Running things

All commands from the repo root (`trading-new/`), using the project venv:

```
# daily live signal (normally launchd, 08:00 local -- run by hand to check now)
./venv/bin/python strategies/five_day_bounce/tech_level_continuation_live.py

# what does the watchlist look like right now
./venv/bin/python strategies/five_day_bounce/tech_level_watchlist.py

# re-run the backtest / OOS confirmation / sensitivity checks
./venv/bin/python strategies/five_day_bounce/tech_level_naive_strategy.py
./venv/bin/python strategies/five_day_bounce/tech_level_oos_strategy.py
./venv/bin/python strategies/five_day_bounce/tech_level_sensitivity.py
./venv/bin/python strategies/five_day_bounce/tech_level_causal_check.py
```

## Known caveats (read before trusting a signal)

- **This isn't actually about technical levels -- found 2026-09-15, and
  it's the most important thing in this file.** A placebo that drops the
  two-touch level/band requirement entirely and keeps only "price is within
  1% of a recently confirmed local low" matches -- and on both universes
  slightly beats -- the real technical-level trades (`ticker_list`: real
  +0.52% vs placebo +0.56%; `oos`: real +0.57% vs placebo +0.65%; see
  `tech_levels_notes.md` 2026-09-15). The two-touch band-matching this
  strategy is built around adds no measurable edge over generic
  short-horizon reversal off a recent local low. The underlying effect is
  still real (see next bullet) but "Five-Day Bounce" and its levels-based
  premise describe the wrong mechanism -- this is the second, larger
  reframing of this effect (first: 2026-09-07, "support holds" ->
  "young-level continuation"; now: "young-level continuation" -> "generic
  local-low reversal, levels optional"). Whether to keep the live rule's
  level requirement (no worse, but no better either, and it cuts the
  tradeable signal count) or widen it to any confirmed local low is an open
  decision -- not made here.
- **Provisional levels -- quantified 2026-09-15, now covering entry AND
  exit, and it's material.** Every level young enough to qualify (<5
  calendar days) is younger than the combo's `distance=10` session
  parameter, so the backtest's whole-history `find_touches` call may
  confirm a touch (support *or* resistance) with future price action a live
  trader wouldn't have had at the time. Fully causal recheck
  (`tech_level_causal_check.py`, `tech_levels_notes.md` 2026-09-15):
  **only ~32% of logged young-level trades would actually have fired using
  only data available at entry time**; of those, the true point-in-time
  exit differs from the hindsight one in just 1.4-1.5% of cases (the exit
  side was a real gap but a minor one -- provably so, since
  `hold_days`=5 < `distance`=10 means no resistance level unconfirmed at
  entry could become confirmable before the hold ends). Fully corrected
  mean excess: **+0.52% (t=3.32, n=344) on `ticker_list`, +0.57% (t=3.88,
  n=493) on `oos`** -- roughly half the reported magnitude vs. the
  full-history backtest (+1.04%/+0.98%, t=10.43/13.35, n=1068/1568). This
  corrected number **does** survive non-i.i.d.-aware inference -- a
  calendar-year block bootstrap 95% CI of [+0.23%, +0.82%] / [+0.34%,
  +0.83%], 0% of 2,000 bootstrap draws <= 0 on either universe, and a
  ticker-clustered t of 3.21/3.54. Treat the published +0.41%/+0.27%
  (t=8.18/4.20) backtest numbers as an optimistic upper bound, not the
  number to size a trade on -- +0.5-0.6% (inference-robust) is the number
  that's actually survived scrutiny so far. Every live buy row still logs
  `provisional` (age in *sessions* <= 10, essentially always True by
  construction) so this can keep being checked against the forward record,
  not just the backtest; the watchlist flags it as `[P]` / a `provisional`
  note.
- **No stop-loss.** The live rule only exits on `hold_days` or a resistance
  touch, never on the trade going wrong. `stop_low` tested as a real
  improvement but isn't wired into `simulate()` or the live script yet.
- **Detection lag** is small (median 1 trading day, max 4, on measured
  history) but real -- a live "N-day-old level" signal is shifted slightly
  later than the same backtest label.
- Backtest numbers assume 10bp round-trip costs, no capital-gains tax, and
  (for the compounded return figures in the artifact) 100% capital
  redeployed into every trade with zero overlap -- unrealistic with a ~7-day
  hold; the measured concurrent-position distribution (median 5, p90 10, max
  24 across the 101-name universe) is in the artifact, step 11, **but that
  figure was measured on the full, hindsight-inflated, all-ages trade set.**
  A concurrency-capped portfolio sim on the much smaller causally-confirmed
  young-level sample (`tech_levels_notes.md` 2026-09-15, Step D) shows
  capacity is essentially never binding at 5-10 slots (344/344 and 493/493
  or 487/493 trades accepted) -- realistic account-level CAGR at 10 slots
  is +4.46%/Sharpe 1.94 (`ticker_list`) and +6.99%/Sharpe 2.14 (`oos`) over
  the ~11-year sample, mostly diluted by idle capital rather than
  capacity-constrained.
- **Fresh-cross-section budget is spent on this question.**
  `config.oos_ticker_list` was already used to confirm the level-age effect
  on 2026-09-07 and again for all four checks above on 2026-09-15 -- it is
  a same-dates confirmation universe for this strategy now, not a fresh
  cross-section; a positive result on it is not independent evidence the
  way a never-touched universe would be. Survivorship bias is also
  unaddressed: both `ticker_list` and `oos_ticker_list` are static,
  present-day large-cap lists with no delisted names ever entering either
  universe.
