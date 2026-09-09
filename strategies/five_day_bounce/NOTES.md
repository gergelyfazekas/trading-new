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
```

## Known caveats (read before trusting a signal)

- **Provisional levels.** Every level young enough to qualify (<5 calendar
  days) is younger than the combo's `distance=10` session parameter -- the
  same zone where `tech_level_live.py`'s `provisional_mask` found a real
  false-start rate (an AAPL band appeared for 5 sessions, then was un-born).
  A live buy can, in principle, later turn out to have been triggered by a
  touch that further price action retroactively un-confirms. Every buy row
  logs `provisional` (age in *sessions* <= 10) so this can be checked, not
  just assumed; the watchlist flags it as `[P]` / a `provisional` note.
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
  24 across the 101-name universe) is in the artifact, step 11.
