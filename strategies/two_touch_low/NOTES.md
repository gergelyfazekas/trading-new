# two_touch_low -- folder map

Live trading infrastructure for the two-touch-low candidate's
`rebound_retest` branch. The research (premise, every variant tried, the
gap=3 mirage, and the full-101-ticker validation numbers) lives in
`mean_reversion_notes.md` at the repo root -- this file only maps what's
in this folder.

**Status (2026-09-22, updated same day): a promising lead put into
forward-record live trading, not a guaranteed edge.** t=+2.40 on 567
trades across the full 101-ticker universe with the market-drawdown exit
now included (t=+2.06 without it); replicated on the 81 tickers never
touched while tuning params, retest_gap, or the drawdown threshold
(t=+2.13 with, +1.99 without). Supersedes an earlier n=535/t=+1.80
figure that used a since-corrected busy-tracking bug (a `flat` signal
blocking a `rebound_retest` slot, which doesn't match what this script
actually does). See `two_touch_low_live.py`'s module docstring and
`mean_reversion_notes.md`'s "Extreme-trade inspection and risk overlays"
section for the full trail, including why near-close runs may under-fire
on this signal's volume gate.

## Files

- `two_touch_low_live.py` -- the live decision engine. `--near-close` (run
  ~21:40 local) or default (run after the prior close settles, ~08:00).
  Pulls SPY alongside the 101 tickers each run for the market-drawdown
  exit (`DD_PCT=0.01`). Same position/log/notification conventions as
  `strategies/five_day_bounce/tech_level_continuation_live.py`.
- `run_live_log_near_close.sh` -- retry/timeout wrapper for the near-close
  run. Not yet wired to a launchd plist.
- `run_live_log.sh` -- retry/timeout wrapper for the settled-close run
  (price-only correction of the prior evening's estimate). Not yet wired
  to a launchd plist.
- `data/positions.json` -- current open positions, freely overwritten.
- `data/signal_log.csv` -- append-only, one row per (run_date, ticker).
  Never regenerate from history -- see `two_touch_low_live.py`'s
  `append_rows` docstring.
- `data/trades.csv` -- append-only closed-trade ledger, the actual forward
  P&L record to judge this against.

## What's deliberately not here

No support/resistance level building (`tech_levels.py`) -- this signal is
a pure trailing price/volume pattern, no persistent level state to
maintain. No early exit on resistance -- there's no resistance band to
exit against, only the market-drawdown exit (below) and the 5-session
hold. No below-L stop-loss -- tested at multiple buffer sizes and
rejected, it made results worse at every size tried (see
`mean_reversion_notes.md`). The `flat` branch of the same rule is not
traded here (see module docstring: it showed no edge at any parameter
tested).
