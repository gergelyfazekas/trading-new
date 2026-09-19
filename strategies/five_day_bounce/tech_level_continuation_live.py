"""Daily live signal generator for the short-horizon continuation effect
found in tech_levels_notes.md (2026-09-07 section): buy within 1% above a
support level that is (a) not yet broken (tech_levels.mark_broken) and (b)
born within the last 5 calendar days; sell after 5 trading days held or on
touching the nearest resistance active at entry, whichever comes first.

Uses the fixed a-priori combo only (data/live_combos.json's "fixed" arm:
distance=10, prominence=0.01, tech_width=0.008) -- frozen exactly as
validated, not re-tuned before going live. Re-tuning against the backtest
right before trading it live would reopen exactly the selection-bias problem
the fixed arm exists to avoid.

Replaces tech_level_live.py's daily cron target: the level-age finding shows
tech_level_live.py's question ("does a support/resistance band predict a
bounce at all") was underpowered relative to the much narrower, stronger
effect here. tech_level_live.py's download/retry/settle-check plumbing
(pull_all, bar_is_settled, _sessions_since) is reused directly rather than
duplicated. Its own daily report loop, data/live_log.csv, and
data/live_combos.json are left untouched as historical record -- not
deleted, not extended further.

IMPORTANT, read before trusting a signal: every level young enough to
qualify here (<=5 calendar days old) is younger than the fixed combo's
`distance=10` session parameter -- exactly the "provisional" zone
tech_level_live.py's provisional_mask docstring already found real (an AAPL
band appeared for 5 sessions, then was un-born once more price data arrived
and n_levels dropped 105->104). A buy signal here can, in principle, later
turn out to have been triggered by a touch that further price action
retroactively un-confirms. This isn't a bug to fix -- it's the unavoidable
shape of trading a young-level effect in real time -- but it does mean the
backtest numbers in tech_levels_notes.md are likely optimistic: they used
touches confirmed with the full benefit of history, so they only ever saw
touches that turned out to be real, never the false starts a live reader
also has to sit through. Going live is the honest fix for this, not a
shortcut around it: a day-by-day system can only ever act on what's
currently detectable, so its forward record is automatically free of that
hindsight bias. Every buy row is logged with provisional=True/False
(age in *sessions* <= distance) so this can be checked later, not just
asserted.

Three files this maintains, with different mutability:
  - data/continuation_positions.json: current open-position state, freely
    overwritten each run -- operational, not a historical record.
  - data/continuation_signal_log.csv: append-only, one row per (run_date,
    ticker). NEVER regenerate this from history -- recomputing an old date
    with today's fuller history would silently convert an out-of-sample
    decision into an in-sample one, same reason data/live_log.csv can't be
    rebuilt (tech_level_live.py's append_log docstring).
  - data/continuation_trades.csv: append-only ledger of closed trades -- the
    actual forward P&L record to judge this against.

Two ways to run this, sharing the same position/log state:

  default (no args) -- as before: run once after the prior US close settles
  (08:00 CEST, via run_live_log.sh), pulling the now-final close for every
  ticker and deciding buy/sell/hold/watch from it.

  --near-close -- added 2026-09-16, scheduled at 21:35 Europe/Budapest
  (~15:35 ET, 25 min before the 16:00 ET close; moved from 21:40 on
  2026-09-19) via run_live_log_near_close.sh. Uses the still-forming intraday bar as a
  stand-in for that day's close, per the user's 2026-09-08 execution-timing
  decision (tech_levels_notes.md, "Execution timing decision"): trading near
  the close you can already see forming is closer to the validated
  same-close backtest behavior than waiting a full session for it to settle
  next morning. Refuses to run outside a 15:15-16:10 ET window
  (`in_near_close_window`) so a misfired or manually re-run invocation can't
  log a mid-session price as if it were near the close. Every row this mode
  writes is flagged `price_estimated=True`.

  The next default (morning) run then looks for that day's
  `price_estimated` row for each ticker and, instead of re-evaluating
  buy/sell/hold from scratch -- which would double-process the same trading
  day against the same continuation_positions.json state (e.g. counting
  days_held twice, or re-triggering a buy already taken the evening before)
  -- just corrects that row's `price` to the now-settled close and clears
  the flag. The decision itself (event, entry/exit, any position mutation)
  already happened at 21:35 and is never redone. If the evening run didn't
  produce an estimate for a ticker (network failure, or the window guard
  tripped), the morning run falls through to the original full evaluation
  for that ticker exactly as before -- the estimate is a same-day head
  start, not a dependency the morning run needs.

Run daily (same slot as tech_level_live.py before it, after the prior US
close), via strategies/five_day_bounce/run_live_log.sh from repo root, or
directly: ./venv/bin/python strategies/five_day_bounce/tech_level_continuation_live.py

Part of the strategies/five_day_bounce package -- see NOTES.md in this folder
for the full strategy writeup and how these scripts fit together.
tech_level_live.py and tech_levels.py stay at the repo root: the former is
shared plumbing for a retired daily report (still used elsewhere), the latter
is the core level-mechanics module several unrelated research scripts import.
"""
import datetime
import json
import os
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from tech_level_live import pull_all, bar_is_settled, SOCKET_TIMEOUT, EXIT_UNSETTLED, _sessions_since
from tech_level_naive_strategy import load_fixed_combo, build_levels, active_support_resistance

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "continuation_positions.json")
SIGNAL_LOG_FILE = os.path.join(os.path.dirname(__file__), "data", "continuation_signal_log.csv")
TRADES_FILE = os.path.join(os.path.dirname(__file__), "data", "continuation_trades.csv")

NEAR_PCT = 0.01
HOLD_DAYS = 5
MAX_AGE_DAYS = 5  # calendar days since birth -- see tech_levels_notes.md, 2026-09-07

# --near-close window, US Eastern. Close is 16:00 ET; 21:35 Europe/Budapest
# lands at ~15:35 ET on both sides of DST (the US and EU shift in different
# weeks, but never by more than the slack built into this window). Anything
# outside 15:15-16:10 ET is refused -- see in_near_close_window. The window
# itself is unchanged by the 21:40->21:35 schedule move (2026-09-19); 15:35 ET
# still sits comfortably inside it.
NEAR_CLOSE_START = datetime.time(15, 15)
NEAR_CLOSE_END = datetime.time(16, 10)
EXIT_OUT_OF_WINDOW = 4  # --near-close run outside NEAR_CLOSE_START/END -- abort, do NOT retry

# Live execution merges config.ticker_list and config.oos_ticker_list (100
# names, order preserved, de-duplicated). The oos reservation ("nothing may
# be used to construct a signal or choose a threshold") is about backtesting
# on that universe's *history* -- it doesn't apply to trading its *future*
# with a rule that's already frozen, since nothing about tomorrow's price
# action could have leaked into a combo/threshold fixed before it happened.
# config.ticker_list/config.oos_ticker_list themselves are left untouched --
# other modules (forecast.py, alpha.py) still rely on their original meaning.
LIVE_TICKERS = list(dict.fromkeys(config.ticker_list + config.oos_ticker_list))

SIGNAL_LOG_COLUMNS = [
    'run_date', 'as_of', 'ticker', 'price', 'price_estimated', 'event', 'reason',
    'support_low', 'support_high', 'support_birth', 'support_age_days', 'provisional', 'bounce_1d',
    'resist_low', 'resist_high',
    'entry_date', 'entry_price', 'exit_price', 'ret', 'days_held',
]
TRADE_COLUMNS = ['ticker', 'entry_date', 'entry_price', 'exit_date', 'exit_price',
                  'exit_reason', 'ret', 'days_held']


def load_positions(path=POSITIONS_FILE):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_positions(positions, path=POSITIONS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(positions, f, indent=2, default=str)


def append_rows(rows, path, columns, key_cols):
    """Upsert `rows` into `path`, keyed on `key_cols` -- same convention as
    tech_level_live.py's append_log, and for the same reason: a retry after a
    partial failure, or a manual rerun, must overwrite that key's row rather
    than duplicate it. Never used to backfill a past date's key with today's
    fuller information (see module docstring).
    """
    if not rows:
        return
    new = pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new_keys = set(map(tuple, new[key_cols].astype(str).values))
        old_keys = old[key_cols].astype(str).apply(tuple, axis=1)
        combined = pd.concat([old[~old_keys.isin(new_keys)], new], ignore_index=True)
    else:
        combined = new
    combined = combined.reindex(columns=columns)
    combined.to_csv(path, index=False)


def bounce_1d_after_birth(close, support):
    """Next trading day's return after a support level's birth date (the
    2nd confirming touch, itself a local trough) -- e.g. birth close 100,
    next close 101 -> 0.01. Diagnostic only: logged for reference on every
    row that has a support, never read by the buy/sell decision anywhere in
    this file. See tech_levels_notes.md, 2026-09-19 "post-birth bounce"
    section -- backtest found a SMALLER post-birth bounce empirically
    preceded better subsequent trades on both ticker_list and oos (the
    opposite of "bigger bounce = more conviction"), but explicitly NOT
    validated enough to gate on: a bounce_1d<median filter would have cut
    total realized CAGR roughly in half at every capital-slot count tested,
    since it discards half of a population whose "worse" half was still
    solidly profitable. Shown on the watchlist purely as an experimental
    reference value -- see tech_level_watchlist.py.

    Returns None if the level was born on the last available bar (no next
    day to measure yet) -- most commonly happens for a level born the same
    session as today's close.
    """
    if support is None:
        return None
    birth_ts = pd.Timestamp(support.birth_date)
    if birth_ts not in close.index:
        return None
    pos = close.index.get_loc(birth_ts)
    if pos + 1 >= len(close):
        return None
    birth_price, next_price = close.iloc[pos], close.iloc[pos + 1]
    if birth_price == 0 or pd.isna(birth_price) or pd.isna(next_price):
        return None
    return float(next_price / birth_price - 1.0)


def _row(as_of, ticker, price, event, reason=None, support=None, age_days=None, provisional=None,
         resistance=None, entry_date=None, entry_price=None, exit_price=None, ret=None, days_held=None,
         price_estimated=False, bounce_1d=None):
    """`resistance` accepts either a Level object (has `.band`, from
    active_support_resistance at entry time) or a plain (low, high) pair (the
    resist_band tuple stashed in `positions[ticker]` for an open position) --
    the held/sell branches below only have the latter, since the Level object
    itself isn't persisted across runs in continuation_positions.json.

    `price_estimated` marks a row logged by a --near-close run, where `price`
    is the still-forming intraday bar rather than the settled close -- see
    module docstring. The next default run corrects it in place.
    """
    resist_band = getattr(resistance, 'band', resistance)
    return {
        'run_date': datetime.date.today(), 'as_of': pd.Timestamp(as_of).date(), 'ticker': ticker,
        'price': round(price, 4), 'price_estimated': price_estimated, 'event': event, 'reason': reason,
        'support_low': round(support.band[0], 4) if support else None,
        'support_high': round(support.band[1], 4) if support else None,
        'support_birth': pd.Timestamp(support.birth_date).date() if support else None,
        'support_age_days': age_days,
        'provisional': provisional,
        'bounce_1d': round(bounce_1d, 4) if bounce_1d is not None else None,
        'resist_low': round(resist_band[0], 4) if resist_band else None,
        'resist_high': round(resist_band[1], 4) if resist_band else None,
        'entry_date': entry_date, 'entry_price': entry_price, 'exit_price': exit_price,
        'ret': ret, 'days_held': days_held,
    }


def evaluate_ticker(ticker, close, combo, positions, distance, price_estimated=False):
    """One (ticker, today) decision. Mutates `positions` in place for this
    ticker. Returns (signal_row, trade_row_or_None).

    `price_estimated` is threaded straight into every _row() call below --
    True from a --near-close run, so the row is flagged for the next default
    run's price-only correction pass (see module docstring and main()).
    """
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    levels = build_levels(close, combo)
    as_of = close.index[-1]
    price = float(close.iloc[-1])
    bar_index = pd.DatetimeIndex(pd.to_datetime(close.index))

    pos = positions.get(ticker)

    if pos is not None:
        days_held = pos['days_held'] + 1
        resist = pos.get('resist_band')
        pos_bounce_1d = pos.get('bounce_1d')  # carried from the original buy -- see below
        hit_resistance = resist is not None and resist[0] <= price <= resist[1]
        if hit_resistance or days_held >= HOLD_DAYS:
            reason = 'resistance' if hit_resistance else 'hold_days'
            ret = price / pos['entry_price'] - 1.0
            del positions[ticker]
            signal_row = _row(as_of, ticker, price, 'sell', reason=reason, resistance=resist,
                               entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                               exit_price=price, ret=ret, days_held=days_held,
                               price_estimated=price_estimated, bounce_1d=pos_bounce_1d)
            trade_row = {
                'ticker': ticker, 'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
                'exit_date': str(pd.Timestamp(as_of).date()), 'exit_price': price,
                'exit_reason': reason, 'ret': ret, 'days_held': days_held,
            }
            return signal_row, trade_row
        else:
            positions[ticker]['days_held'] = days_held
            signal_row = _row(as_of, ticker, price, 'held', resistance=resist,
                               entry_date=pos['entry_date'], entry_price=pos['entry_price'], days_held=days_held,
                               price_estimated=price_estimated, bounce_1d=pos_bounce_1d)
            return signal_row, None

    support, resistance = active_support_resistance(levels, as_of, price)
    age_days = provisional = bounce_1d = None
    if support is not None:
        age_days = (pd.Timestamp(as_of) - pd.Timestamp(support.birth_date)).days
        age_sessions = _sessions_since(bar_index, support.birth_date, as_of)
        provisional = age_sessions is not None and age_sessions <= distance
        bounce_1d = bounce_1d_after_birth(close, support)
        dist = (price - support.band[1]) / price
        if age_days < MAX_AGE_DAYS and 0 <= dist <= NEAR_PCT:
            positions[ticker] = {
                'entry_date': str(pd.Timestamp(as_of).date()), 'entry_price': price, 'days_held': 0,
                'resist_band': list(resistance.band) if resistance is not None else None,
                'bounce_1d': bounce_1d,
            }
            signal_row = _row(as_of, ticker, price, 'buy', support=support, age_days=age_days,
                               provisional=provisional, resistance=resistance,
                               price_estimated=price_estimated, bounce_1d=bounce_1d)
            return signal_row, None

    signal_row = _row(as_of, ticker, price, 'watch', support=support, age_days=age_days,
                       provisional=provisional, resistance=resistance,
                       price_estimated=price_estimated, bounce_1d=bounce_1d)
    return signal_row, None


def in_near_close_window(now=None):
    """True if the current US Eastern time is inside NEAR_CLOSE_START/END --
    close enough to the 16:00 ET close that the live price is a reasonable
    stand-in for it (see module docstring, "--near-close"). Guards against a
    misfired or manually re-run --near-close invocation logging a mid-session
    price as if it were near the close.
    """
    et = ZoneInfo("America/New_York")
    now = now or datetime.datetime.now(et)
    return NEAR_CLOSE_START <= now.time() <= NEAR_CLOSE_END


def _load_existing_log(path=SIGNAL_LOG_FILE):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["run_date", "as_of"])


def _correct_estimate(existing_log, ticker, as_of, price):
    """If `existing_log` already carries a --near-close estimate for (as_of,
    ticker), return a corrected copy of that row with `price` set to the now-
    settled close and `price_estimated` cleared -- nothing else changes,
    since the decision it recorded (event, entry/exit, any position
    mutation) already happened that evening and must not be redone. Returns
    None if there's no matching estimate, so the caller falls through to a
    full evaluate_ticker() call exactly as before.
    """
    if existing_log is None or "price_estimated" not in existing_log.columns:
        return None
    as_of_date = pd.Timestamp(as_of).date()
    match = existing_log[
        (existing_log["ticker"] == ticker)
        & (existing_log["as_of"].dt.date == as_of_date)
        & (existing_log["price_estimated"] == True)  # noqa: E712
    ]
    if match.empty:
        return None
    row = match.iloc[-1].to_dict()
    row["run_date"] = datetime.date.today()
    row["as_of"] = as_of_date
    row["price"] = round(price, 4)
    row["price_estimated"] = False
    return row


def _already_logged(existing_log, ticker, as_of):
    """True if `existing_log` already has any row for (ticker, as_of), from
    any prior run (near-close estimate, already-corrected, or a plain
    default evaluation). A settled day that already has a row must never be
    silently re-derived by a later run -- see the module docstring's
    append-only invariant for continuation_signal_log.csv.

    Without this check, if `_correct_estimate` fails to match an evening's
    price_estimated=True row for any reason (a missed run, a stale read, a
    manual out-of-band invocation), the code falls through to a fresh
    evaluate_ticker() call for the same (as_of, ticker) key. Since the
    ticker is already in `positions`, that produces a 'held' row -- and
    append_rows' upsert (keyed on as_of+ticker) then silently overwrites the
    original 'buy' row with it, erasing the only record that a buy signal
    ever fired even though the position (and the money) is real. Confirmed
    happening in continuation_signal_log.csv (MSFT 2026-09-17, HON/ABT
    2026-09-15) -- see tech_levels_notes.md.
    """
    if existing_log is None:
        return False
    as_of_date = pd.Timestamp(as_of).date()
    return bool((
        (existing_log["ticker"] == ticker)
        & (existing_log["as_of"].dt.date == as_of_date)
    ).any())


def _notify_near_close(buys, missing_count):
    """Best-effort macOS notification for a --near-close run -- the whole
    point of running this before close is to see it in the moment, not to
    read a log file afterward. Never raises: a notification failure must not
    fail the run or skip logging what was actually decided.
    """
    if buys:
        title = f"BUY {len(buys)} at close"
        message = ", ".join(sorted(r['ticker'] for r in buys))
    else:
        title = "Tech Level Continuation"
        message = "Nothing to buy at close today"
    if missing_count:
        message += f" ({missing_count} ticker(s) missing)"
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(message)} with title {json.dumps(title)} sound name "Glass"'],
            timeout=10, check=False,
        )
    except Exception:
        pass


def main():
    socket.setdefaulttimeout(SOCKET_TIMEOUT)

    near_close = "--near-close" in sys.argv[1:]
    if near_close and not in_near_close_window():
        print("ABORTED: --near-close run outside the 15:15-16:10 ET pre-close window; "
              "refusing to log a mid-session price as a close estimate. Tomorrow's "
              "default run will still catch this day via the normal full evaluation.")
        return EXIT_OUT_OF_WINDOW

    combo = load_fixed_combo()
    distance = combo['distance']
    tickers = LIVE_TICKERS

    series, failed = pull_all(tickers)
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}\n")

    positions = load_positions()
    existing_log = None if near_close else _load_existing_log()

    signal_rows, trade_rows, corrected, already_logged, unsettled, errored = [], [], [], [], [], []
    for ticker in tickers:
        if ticker not in series:
            continue
        close, _volume = series[ticker]
        as_of = close.index[-1]
        price = float(close.iloc[-1])

        if near_close:
            et_today = datetime.datetime.now(ZoneInfo("America/New_York")).date()
            if pd.Timestamp(as_of).date() != et_today:
                unsettled.append(ticker)
                continue
        else:
            if not bar_is_settled(as_of):
                unsettled.append(ticker)
                continue
            corrected_row = _correct_estimate(existing_log, ticker, as_of, price)
            if corrected_row is not None:
                signal_rows.append(corrected_row)
                corrected.append(ticker)
                continue
            if _already_logged(existing_log, ticker, as_of):
                already_logged.append(ticker)
                continue

        try:
            signal_row, trade_row = evaluate_ticker(ticker, close, combo, positions, distance,
                                                      price_estimated=near_close)
            signal_rows.append(signal_row)
            if trade_row is not None:
                trade_rows.append(trade_row)
        except Exception as exc:
            errored.append(ticker)
            print(f"{ticker}: SKIPPED -- {exc.__class__.__name__}: {exc}")

    if unsettled:
        if near_close:
            print(f"ABORTED: {len(unsettled)} ticker(s) returned a bar not dated today "
                  f"({', '.join(sorted(unsettled)[:5])}...) -- stale data or market closed today. "
                  f"Nothing logged, positions unchanged. Tomorrow's default run will still catch "
                  f"today normally.")
        else:
            print(f"ABORTED: latest bar for {len(unsettled)} ticker(s) has not settled "
                  f"({', '.join(sorted(unsettled)[:5])}...). Nothing logged, positions unchanged.")
        return EXIT_UNSETTLED

    save_positions(positions)
    append_rows(signal_rows, SIGNAL_LOG_FILE, SIGNAL_LOG_COLUMNS, key_cols=['as_of', 'ticker'])
    append_rows(trade_rows, TRADES_FILE, TRADE_COLUMNS, key_cols=['ticker', 'entry_date'])

    buys = [r for r in signal_rows if r['event'] == 'buy']
    sells = [r for r in signal_rows if r['event'] == 'sell']
    held = [r for r in signal_rows if r['event'] == 'held']

    print(f"\n{'='*60}")
    if near_close:
        now_et = datetime.datetime.now(ZoneInfo("America/New_York")).strftime('%H:%M')
        print(f"NEAR-CLOSE ESTIMATE -- {datetime.date.today()} {now_et} ET")
        print("price is a same-day estimate (run ~20 min before close); tomorrow's")
        print("default run corrects it to the settled close -- the decision below is final")
    else:
        print(f"=== {datetime.date.today()} ===")
    print('='*60)
    if buys:
        print("BUY:")
        for r in buys:
            flag = " [PROVISIONAL -- could still be revised, see module docstring]" if r['provisional'] else ""
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  support [{r['support_low']:.2f}, {r['support_high']:.2f}] "
                  f"born {r['support_birth']} ({r['support_age_days']}d old){flag}")
    elif near_close:
        print("NOTHING TO BUY today.")
    if sells:
        print("SELL:")
        for r in sells:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  entry {r['entry_price']:.2f} on {r['entry_date']} "
                  f"-> {r['ret']:+.2%}  ({r['reason']}, held {r['days_held']}d)")
    if held:
        print(f"HELD ({len(held)}): {', '.join(sorted(r['ticker'] for r in held))}")
    if corrected:
        print(f"CORRECTED (price only, decision already made last evening): {len(corrected)} ticker(s)")
    if already_logged:
        print(f"ALREADY LOGGED (skipped re-evaluation, today's row exists): {len(already_logged)} ticker(s): "
              f"{', '.join(sorted(already_logged))}")
    if not buys and not sells and not near_close:
        print("no new signals today")

    missing = sorted(set(failed) | set(errored))

    if near_close:
        _notify_near_close(buys, len(missing))

    if missing:
        print(f"\nincomplete: {len(missing)} ticker(s) absent today: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
