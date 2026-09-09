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
import sys

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
    'run_date', 'as_of', 'ticker', 'price', 'event', 'reason',
    'support_low', 'support_high', 'support_birth', 'support_age_days', 'provisional',
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


def _row(as_of, ticker, price, event, reason=None, support=None, age_days=None, provisional=None,
         resistance=None, entry_date=None, entry_price=None, exit_price=None, ret=None, days_held=None):
    """`resistance` accepts either a Level object (has `.band`, from
    active_support_resistance at entry time) or a plain (low, high) pair (the
    resist_band tuple stashed in `positions[ticker]` for an open position) --
    the held/sell branches below only have the latter, since the Level object
    itself isn't persisted across runs in continuation_positions.json.
    """
    resist_band = getattr(resistance, 'band', resistance)
    return {
        'run_date': datetime.date.today(), 'as_of': pd.Timestamp(as_of).date(), 'ticker': ticker,
        'price': round(price, 4), 'event': event, 'reason': reason,
        'support_low': round(support.band[0], 4) if support else None,
        'support_high': round(support.band[1], 4) if support else None,
        'support_birth': pd.Timestamp(support.birth_date).date() if support else None,
        'support_age_days': age_days,
        'provisional': provisional,
        'resist_low': round(resist_band[0], 4) if resist_band else None,
        'resist_high': round(resist_band[1], 4) if resist_band else None,
        'entry_date': entry_date, 'entry_price': entry_price, 'exit_price': exit_price,
        'ret': ret, 'days_held': days_held,
    }


def evaluate_ticker(ticker, close, combo, positions, distance):
    """One (ticker, today) decision. Mutates `positions` in place for this
    ticker. Returns (signal_row, trade_row_or_None).
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
        hit_resistance = resist is not None and resist[0] <= price <= resist[1]
        if hit_resistance or days_held >= HOLD_DAYS:
            reason = 'resistance' if hit_resistance else 'hold_days'
            ret = price / pos['entry_price'] - 1.0
            del positions[ticker]
            signal_row = _row(as_of, ticker, price, 'sell', reason=reason, resistance=resist,
                               entry_date=pos['entry_date'], entry_price=pos['entry_price'],
                               exit_price=price, ret=ret, days_held=days_held)
            trade_row = {
                'ticker': ticker, 'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
                'exit_date': str(pd.Timestamp(as_of).date()), 'exit_price': price,
                'exit_reason': reason, 'ret': ret, 'days_held': days_held,
            }
            return signal_row, trade_row
        else:
            positions[ticker]['days_held'] = days_held
            signal_row = _row(as_of, ticker, price, 'held', resistance=resist,
                               entry_date=pos['entry_date'], entry_price=pos['entry_price'], days_held=days_held)
            return signal_row, None

    support, resistance = active_support_resistance(levels, as_of, price)
    age_days = provisional = None
    if support is not None:
        age_days = (pd.Timestamp(as_of) - pd.Timestamp(support.birth_date)).days
        age_sessions = _sessions_since(bar_index, support.birth_date, as_of)
        provisional = age_sessions is not None and age_sessions <= distance
        dist = (price - support.band[1]) / price
        if age_days < MAX_AGE_DAYS and 0 <= dist <= NEAR_PCT:
            positions[ticker] = {
                'entry_date': str(pd.Timestamp(as_of).date()), 'entry_price': price, 'days_held': 0,
                'resist_band': list(resistance.band) if resistance is not None else None,
            }
            signal_row = _row(as_of, ticker, price, 'buy', support=support, age_days=age_days,
                               provisional=provisional, resistance=resistance)
            return signal_row, None

    signal_row = _row(as_of, ticker, price, 'watch', support=support, age_days=age_days,
                       provisional=provisional, resistance=resistance)
    return signal_row, None


def main():
    socket.setdefaulttimeout(SOCKET_TIMEOUT)

    combo = load_fixed_combo()
    distance = combo['distance']
    tickers = LIVE_TICKERS

    series, failed = pull_all(tickers)
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}\n")

    positions = load_positions()

    signal_rows, trade_rows, unsettled, errored = [], [], [], []
    for ticker in tickers:
        if ticker not in series:
            continue
        close, _volume = series[ticker]
        as_of = close.index[-1]
        if not bar_is_settled(as_of):
            unsettled.append(ticker)
            continue
        try:
            signal_row, trade_row = evaluate_ticker(ticker, close, combo, positions, distance)
            signal_rows.append(signal_row)
            if trade_row is not None:
                trade_rows.append(trade_row)
        except Exception as exc:
            errored.append(ticker)
            print(f"{ticker}: SKIPPED -- {exc.__class__.__name__}: {exc}")

    if unsettled:
        print(f"ABORTED: latest bar for {len(unsettled)} ticker(s) has not settled "
              f"({', '.join(sorted(unsettled)[:5])}...). Nothing logged, positions unchanged.")
        return EXIT_UNSETTLED

    save_positions(positions)
    append_rows(signal_rows, SIGNAL_LOG_FILE, SIGNAL_LOG_COLUMNS, key_cols=['as_of', 'ticker'])
    append_rows(trade_rows, TRADES_FILE, TRADE_COLUMNS, key_cols=['ticker', 'entry_date'])

    buys = [r for r in signal_rows if r['event'] == 'buy']
    sells = [r for r in signal_rows if r['event'] == 'sell']
    held = [r for r in signal_rows if r['event'] == 'held']

    print(f"\n=== {datetime.date.today()} ===")
    if buys:
        print("BUY:")
        for r in buys:
            flag = " [PROVISIONAL -- could still be revised, see module docstring]" if r['provisional'] else ""
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  support [{r['support_low']:.2f}, {r['support_high']:.2f}] "
                  f"born {r['support_birth']} ({r['support_age_days']}d old){flag}")
    if sells:
        print("SELL:")
        for r in sells:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  entry {r['entry_price']:.2f} on {r['entry_date']} "
                  f"-> {r['ret']:+.2%}  ({r['reason']}, held {r['days_held']}d)")
    if held:
        print(f"HELD ({len(held)}): {', '.join(sorted(r['ticker'] for r in held))}")
    if not buys and not sells:
        print("no new signals today")

    missing = sorted(set(failed) | set(errored))
    if missing:
        print(f"\nincomplete: {len(missing)} ticker(s) absent today: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
