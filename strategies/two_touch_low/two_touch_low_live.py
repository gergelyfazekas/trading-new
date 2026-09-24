"""Daily live signal generator for the two-touch-low candidate's
rebound_retest branch -- mean_reversion_notes.md, "Two-touch low, fully
specified" (2026-09-22).

STATUS, read before trusting a signal from this script: a promising lead
with the strongest validation this candidate has produced, not a
guaranteed edge. On the full 101-ticker LIVE_TICKERS universe (locked
params, retest_gap=2), using the day-by-day causal harness
(two_touch_low_daybyday_dd_exit.py -- the live-matching population, where
a `flat` signal never blocks a `rebound_retest` slot, unlike an earlier
harness that gave a since-retired n=535/t=1.80 figure):

  | | n | hit | excess | t |
  |---|---|---|---|---|
  | 5-day hold only | 566 | 62.9% | +0.28% | +2.06 |
  | + market-drawdown exit (see below) | 567 | 60.1% | +0.28% | +2.40 |

Both halves replicated on the 81 tickers never touched while tuning
params, retest_gap, or the drawdown threshold (5-day-hold-only: t=+1.99;
with the drawdown exit: t=+2.13) -- three consistent confirmations
(in-sample, fresh-81 alone, combined), no reversals. See
mean_reversion_notes.md for the full history, including the retest_gap=3
mirage and the below-L stop-loss variants that were tested and rejected.
The `flat` branch of the same rule showed no edge at any parameter tested
and is deliberately NOT traded here.

Rule (rebound_retest branch only; see two_touch_low_daybyday.py for the
full spec and the flat branch this script skips):
  Stage 1 -- first low at day L: backward lookback-day local min,
    prominence vs L-1, volume_ratio[L] > volume_threshold.
  Stage 2 -- L+1 rebounds between k_pct and rebound_max_pct above L, on
    volume_ratio[L+1] < rebound_volume_threshold (weak volume). L+2 (today,
    for a fresh signal) retests back into [close[L], close[L+1]] on
    volume_ratio[L+2] > volume_threshold_buy.
  Exit: the first of (a) SPY closes DD_PCT (1%) or more below its own
    close on this trade's entry day, checked fresh every session of the
    hold (two_touch_low_daybyday_dd_exit.py's market-drawdown exit -- cuts
    the fat left tail from broad-market crashes landing mid-hold, e.g. the
    2020 COVID and 2023 SVB weeks; doesn't reduce average excess return,
    reduces its variance), or (b) a fixed 5-session hold, whichever comes
    first. A below-L stop-loss (exit if price breaks the original low) was
    also tested and made things worse at every buffer size tried -- not
    used here; see mean_reversion_notes.md.

IMPORTANT volume caveat for --near-close: unlike price, the backtest's
volume_ratio at the retest day (L+2, i.e. "today" for a live buy) was
always computed on a FULLY SETTLED day's volume. A --near-close run at
~15:40 ET uses yfinance's still-forming intraday bar for both price AND
volume, same as the old five-day-bounce live script did for price alone --
but volume tends to be more back-loaded within a session (closing-auction
prints) than price is representative of its own close, so a near-close
volume_ratio understates the settled figure more than near-close price
misrepresents the settled close. Net effect: this script is more likely to
MISS a real signal near-close (false negative) than to fire a false one --
the gate only gets easier to clear as the day's volume completes, not
harder. The next morning's default run corrects price only (same
price_estimated convention as tech_level_continuation_live.py) and does
NOT re-run the volume-dependent decision -- so a signal missed near-close
because of partial volume stays missed. If that miss rate turns out to
matter, the fix is to stop trading near-close for this signal specifically
and rely on the 08:00 settled run instead; not done here since it wasn't
asked for.

Two ways to run this, sharing the same position/log state (mirrors
tech_level_continuation_live.py exactly):
  default (no args) -- run after the prior US close settles (08:00 CEST
  slot); pulls the now-final close/volume and corrects any --near-close
  estimate's price in place, or does a full evaluation for anything that
  doesn't have one.
  --near-close -- run ~21:40 local (~15:40 ET, before the 16:00 ET close),
  flags every row price_estimated=True, sends a macOS notification. Inside
  the 15:15-16:10 ET window this is a real decision (logged, positions
  mutated). Outside it, this becomes a DRY RUN instead of aborting: still
  pulls live data and evaluates/prints every ticker, but never logs,
  never touches positions.json, and never notifies -- safe to call any
  time of day just to see the current read.

Files, same mutability convention as continuation_positions.json /
continuation_signal_log.csv / continuation_trades.csv:
  data/positions.json -- current open-position state, freely overwritten.
  data/signal_log.csv -- append-only, one row per (run_date, ticker). Never
    regenerate from history.
  data/trades.csv -- append-only ledger of closed trades, the forward
    record to judge this against.

Run: ./venv/bin/python strategies/two_touch_low/two_touch_low_live.py [--near-close]
"""
import copy
import datetime
import json
import os
import socket
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "five_day_bounce"))
sys.path.insert(0, os.path.join(HERE, "..", "five_day_bounce", "experiments"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from tech_level_live import pull_all, bar_is_settled, SOCKET_TIMEOUT, EXIT_UNSETTLED
from tech_level_continuation_live import LIVE_TICKERS, NEAR_CLOSE_START, NEAR_CLOSE_END, in_near_close_window
from two_touch_low_daybyday import PARAMS, HOLD, decide, diagnose, STAGE_RANK

RETEST_GAP = 2  # locked spec's second-touch gap -- see mean_reversion_notes.md
DD_PCT = 0.01  # market-drawdown exit threshold -- validated in- and out-of-sample, see module docstring

STAGE_LABEL = {
    "retest_volume_fail": "retest OK, volume too low -- closed",
    "retest_out_of_band": "retest broke the [L, L+1] band -- closed",
    "awaiting_retest_day": "rebound OK, retest is next session",
    "rebound_volume_too_high": "rebound volume too high -- closed",
    "rebound_too_big": "rebound too big -- closed",
    "flat_branch_only": "flat rebound only, not traded live -- closed",
    "no_rebound": "no rebound -- closed",
    "birth_volume_fail": "low-day volume too low -- closed",
    "prominence_fail": "prominence check failed -- closed",
    "awaiting_rebound_day": "low confirmed, rebound check is next session",
}

POSITIONS_FILE = os.path.join(HERE, "data", "positions.json")
SIGNAL_LOG_FILE = os.path.join(HERE, "data", "signal_log.csv")
TRADES_FILE = os.path.join(HERE, "data", "trades.csv")

SIGNAL_LOG_COLUMNS = [
    'run_date', 'as_of', 'ticker', 'price', 'price_estimated', 'event',
    'low_date', 'pct1', 'vr_L', 'vr_L1', 'vr_L2',
    'entry_date', 'entry_price', 'exit_price', 'exit_reason', 'ret', 'days_held',
]
TRADE_COLUMNS = ['ticker', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'exit_reason', 'ret', 'days_held']


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
    """Upsert on key_cols -- same convention as tech_level_continuation_live's
    append_rows, for the same reason: a retry or manual rerun must overwrite
    that key's row, never duplicate it, and this must never be used to
    backfill a past date with fuller present-day information."""
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
    combined.reindex(columns=columns).to_csv(path, index=False)


def _row(as_of, ticker, price, event, low_date=None, pct1=None, vr_L=None, vr_L1=None, vr_L2=None,
         entry_date=None, entry_price=None, exit_price=None, exit_reason=None, ret=None, days_held=None,
         price_estimated=False):
    return {
        'run_date': datetime.date.today(), 'as_of': pd.Timestamp(as_of).date(), 'ticker': ticker,
        'price': round(price, 4), 'price_estimated': price_estimated, 'event': event,
        'low_date': low_date, 'pct1': round(pct1, 4) if pct1 is not None else None,
        'vr_L': round(vr_L, 3) if vr_L is not None else None,
        'vr_L1': round(vr_L1, 3) if vr_L1 is not None else None,
        'vr_L2': round(vr_L2, 3) if vr_L2 is not None else None,
        'entry_date': entry_date, 'entry_price': entry_price, 'exit_price': exit_price,
        'exit_reason': exit_reason, 'ret': ret, 'days_held': days_held,
    }


def evaluate_ticker(ticker, close, volume, spy_price, positions, price_estimated=False):
    """One (ticker, today) decision. Mutates `positions` in place. Returns
    (signal_row, trade_row_or_None). Only ever opens a position on the
    rebound_retest branch -- a `flat`-branch decide() result is ignored,
    same as if it weren't there, since that branch showed no edge (see
    module docstring).

    `spy_price` is today's SPY close (or near-close estimate), used only
    to check the market-drawdown exit against the SPY price recorded at
    this position's own entry (`pos['entry_spy']`) -- never anything from
    a future session."""
    close = close.copy()
    close.index = pd.to_datetime(close.index)
    as_of = close.index[-1]
    price = float(close.iloc[-1])

    pos = positions.get(ticker)
    if pos is not None:
        days_held = pos['days_held'] + 1
        entry_spy = pos.get('entry_spy')
        dd_hit = (entry_spy is not None and spy_price is not None and not pd.isna(spy_price)
                  and spy_price / entry_spy - 1 <= -DD_PCT)
        if dd_hit or days_held >= HOLD:
            reason = 'market_dd' if dd_hit and days_held < HOLD else 'hold_days'
            ret = price / pos['entry_price'] - 1.0
            del positions[ticker]
            signal_row = _row(as_of, ticker, price, 'sell', entry_date=pos['entry_date'],
                               entry_price=pos['entry_price'], exit_price=price, exit_reason=reason, ret=ret,
                               days_held=days_held, price_estimated=price_estimated)
            trade_row = {'ticker': ticker, 'entry_date': pos['entry_date'], 'entry_price': pos['entry_price'],
                         'exit_date': str(pd.Timestamp(as_of).date()), 'exit_price': price,
                         'exit_reason': reason, 'ret': ret, 'days_held': days_held}
            return signal_row, trade_row
        positions[ticker]['days_held'] = days_held
        signal_row = _row(as_of, ticker, price, 'held', entry_date=pos['entry_date'],
                           entry_price=pos['entry_price'], days_held=days_held, price_estimated=price_estimated)
        return signal_row, None

    n = len(close)
    L = n - 1 - RETEST_GAP
    decision = decide(close, volume, L, PARAMS, retest_gap=RETEST_GAP) if L >= PARAMS["lookback"] else None
    if decision is not None and decision["branch"] == "rebound_retest":
        positions[ticker] = {'entry_date': str(pd.Timestamp(as_of).date()), 'entry_price': price,
                              'entry_spy': None if spy_price is None or pd.isna(spy_price) else float(spy_price),
                              'days_held': 0}
        signal_row = _row(as_of, ticker, price, 'buy', low_date=str(close.index[L].date()),
                           pct1=decision["pct1"], vr_L=decision["vr_L"], vr_L1=decision["vr_L1"],
                           vr_L2=decision["vr_L2"], price_estimated=price_estimated)
        return signal_row, None

    signal_row = _row(as_of, ticker, price, 'watch', price_estimated=price_estimated)
    return signal_row, None


def _load_existing_log(path=SIGNAL_LOG_FILE):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["run_date", "as_of"])


def _correct_estimate(existing_log, ticker, as_of, price):
    """Same convention as tech_level_continuation_live._correct_estimate:
    price only, never re-decides -- see module docstring's volume caveat
    for why a near-close 'watch'/no-buy can't simply be re-tried here."""
    if existing_log is None:
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
    if existing_log is None:
        return False
    as_of_date = pd.Timestamp(as_of).date()
    return bool(((existing_log["ticker"] == ticker) & (existing_log["as_of"].dt.date == as_of_date)).any())


def _format_diag(d):
    bits = [STAGE_LABEL.get(d['stage'], d['stage'])]
    for k, label in (('pct1', 'pct1'), ('vr_L', 'vr_L'), ('vr_L1', 'vr_L1'), ('vr_L2', 'vr_L2')):
        if k in d:
            v = d[k]
            bits.append(f"{label}={v:+.2%}" if k == 'pct1' else f"{label}={v:.2f}x")
    return ' | '.join(bits)


def _notify_near_close(buys, missing_count):
    if buys:
        title = f"BUY {len(buys)} at close"
        message = ", ".join(sorted(r['ticker'] for r in buys))
    else:
        title = "Two-Touch Low"
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
    # Outside the pre-close window, --near-close no longer aborts -- it
    # becomes a dry run: pull live data, evaluate, print, but never log or
    # notify, and never mutate positions.json. Useful for checking the
    # current read at any time of day without it counting as a real
    # decision. The scheduled 21:40 job normally lands inside the window
    # (real near-close mode); this only kicks in for an ad-hoc/early call
    # or a late-firing scheduled run.
    dry_run = near_close and not in_near_close_window()

    tickers = LIVE_TICKERS
    series, failed = pull_all(tickers + ["SPY"])
    if "SPY" in failed:
        print("warning: no SPY data -- market-drawdown exit disabled this run, positions fall back to the "
              "5-session hold only\n")
    spy_close = None
    if "SPY" in series:
        spy_close, _ = series["SPY"]
        spy_close = spy_close.copy()
        spy_close.index = pd.to_datetime(spy_close.index)
    failed = [f for f in failed if f != "SPY"]
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}\n")

    positions = load_positions()
    # dry run must never persist a position change, so it evaluates against
    # a throwaway copy -- evaluate_ticker() mutates its `positions` arg.
    eval_positions = copy.deepcopy(positions) if dry_run else positions
    existing_log = None if near_close else _load_existing_log()

    signal_rows, trade_rows, corrected, already_logged, unsettled, errored = [], [], [], [], [], []
    for ticker in tickers:
        if ticker not in series:
            continue
        close, volume = series[ticker]
        if volume is None:
            volume = pd.Series(np.nan, index=close.index)
        as_of = close.index[-1]
        price = float(close.iloc[-1])
        # SPY aligned to this ticker's own "today" -- same as_of date, not
        # SPY's own last row, in case the two ever briefly desync (holidays,
        # a late/early print for one of the two).
        spy_price = None
        if spy_close is not None:
            as_of_ts = pd.Timestamp(as_of)
            spy_today = spy_close[spy_close.index <= as_of_ts]
            if len(spy_today) and spy_today.index[-1] == as_of_ts:
                spy_price = float(spy_today.iloc[-1])

        if near_close and not dry_run:
            et_today = datetime.datetime.now(ZoneInfo("America/New_York")).date()
            if pd.Timestamp(as_of).date() != et_today:
                unsettled.append(ticker)
                continue
        elif not near_close:
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
        # dry_run: no date/settle check -- evaluate on whatever the latest
        # available bar is, purely informational.

        try:
            signal_row, trade_row = evaluate_ticker(ticker, close, volume, spy_price, eval_positions,
                                                      price_estimated=near_close)
            if signal_row["event"] == "watch":
                # diagnostic only, never persisted -- append_rows filters to
                # SIGNAL_LOG_COLUMNS so these extra keys are dropped on write.
                n_bars = len(close)
                # today's candidate (low 2 sessions back) -- already fully
                # resolved, explains why today didn't buy.
                signal_row["_diag"] = diagnose(close, volume, n_bars - 1 - RETEST_GAP, PARAMS,
                                                retest_gap=RETEST_GAP)
                # still-developing candidates: yesterday's low (rebound
                # happened today, retest is next session) and today's low
                # (just formed, rebound check is next session).
                signal_row["_diag_fwd"] = [
                    diagnose(close, volume, n_bars - 2, PARAMS, retest_gap=RETEST_GAP),
                    diagnose(close, volume, n_bars - 1, PARAMS, retest_gap=RETEST_GAP),
                ]
            signal_rows.append(signal_row)
            if trade_row is not None:
                trade_rows.append(trade_row)
        except Exception as exc:
            errored.append(ticker)
            print(f"{ticker}: SKIPPED -- {exc.__class__.__name__}: {exc}")

    if unsettled:
        if near_close:
            print(f"ABORTED: {len(unsettled)} ticker(s) returned a bar not dated today "
                  f"({', '.join(sorted(unsettled)[:5])}...). Nothing logged, positions unchanged.")
        else:
            print(f"ABORTED: latest bar for {len(unsettled)} ticker(s) has not settled "
                  f"({', '.join(sorted(unsettled)[:5])}...). Nothing logged, positions unchanged.")
        return EXIT_UNSETTLED

    if not dry_run:
        save_positions(positions)
        append_rows(signal_rows, SIGNAL_LOG_FILE, SIGNAL_LOG_COLUMNS, key_cols=['as_of', 'ticker'])
        append_rows(trade_rows, TRADES_FILE, TRADE_COLUMNS, key_cols=['ticker', 'entry_date'])

    buys = [r for r in signal_rows if r['event'] == 'buy']
    sells = [r for r in signal_rows if r['event'] == 'sell']
    held = [r for r in signal_rows if r['event'] == 'held']

    print(f"\n{'='*60}")
    if dry_run:
        now_et = datetime.datetime.now(ZoneInfo("America/New_York")).strftime('%H:%M')
        print(f"DRY RUN -- {datetime.date.today()} {now_et} ET  (two-touch-low, rebound_retest)")
        print(f"outside the {NEAR_CLOSE_START}-{NEAR_CLOSE_END} ET pre-close window -- "
              f"NOT logged, positions unchanged, no notification")
    elif near_close:
        now_et = datetime.datetime.now(ZoneInfo("America/New_York")).strftime('%H:%M')
        print(f"NEAR-CLOSE ESTIMATE -- {datetime.date.today()} {now_et} ET  (two-touch-low, rebound_retest)")
        print("price/volume are same-day estimates; volume gate may under-fire near-close (see module docstring)")
    else:
        print(f"=== {datetime.date.today()} (two-touch-low, rebound_retest) ===")
    print('='*60)
    if buys:
        print("BUY:")
        for r in buys:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  low {r['low_date']}  pct1={r['pct1']:+.2%}  "
                  f"vr_L={r['vr_L']:.2f}x  vr_L1={r['vr_L1']:.2f}x  vr_L2={r['vr_L2']:.2f}x")
    elif near_close:
        print("NOTHING TO BUY today.")
    if sells:
        print("SELL:")
        for r in sells:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  entry {r['entry_price']:.2f} on {r['entry_date']} "
                  f"-> {r['ret']:+.2%}  (held {r['days_held']}d, {r['exit_reason']})")
    if held:
        print(f"HELD ({len(held)}): {', '.join(sorted(r['ticker'] for r in held))}")

    watch_rows = [r for r in signal_rows if r.get('event') == 'watch' and '_diag' in r]
    boring = {"not_a_low", "insufficient_history"}

    interesting = sorted(
        (r for r in watch_rows if r['_diag']['stage'] not in boring),
        key=lambda r: STAGE_RANK.get(r['_diag']['stage'], 99),
    )
    if interesting:
        print(f"\nTODAY'S CANDIDATE, already closed either way ({len(interesting)} of {len(watch_rows)} watched "
              f"-- {len(watch_rows) - len(interesting)} with no active low 2 sessions ago omitted):")
        for r in interesting:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  {_format_diag(r['_diag'])}")
    elif watch_rows:
        print(f"\n{len(watch_rows)} watched, none had an active low 2 sessions ago.")

    # unlike "interesting" above, this must only include stages that can
    # still change -- everything else is a closed case for that L too,
    # just a different (more recent) one than "today's candidate".
    still_open = {"awaiting_rebound_day", "awaiting_retest_day"}
    fwd_tag = {0: "low 1 session ago", 1: "low today"}
    fwd_rows = sorted(
        ((r, d, fwd_tag[i]) for r in watch_rows for i, d in enumerate(r.get('_diag_fwd', []))
         if d['stage'] in still_open),
        key=lambda x: STAGE_RANK.get(x[1]['stage'], 99),
    )
    if fwd_rows:
        print(f"\nSTILL DEVELOPING, may resolve over the next session or two ({len(fwd_rows)}):")
        for r, d, tag in fwd_rows:
            print(f"  {r['ticker']:6} @ {r['price']:.2f}  ({tag})  {_format_diag(d)}")

    if corrected:
        print(f"CORRECTED (price only): {len(corrected)} ticker(s)")
    if already_logged:
        print(f"ALREADY LOGGED (skipped): {len(already_logged)} ticker(s)")
    if not buys and not sells and not near_close:
        print("no new signals today")

    missing = sorted(set(failed) | set(errored))
    if near_close and not dry_run:
        _notify_near_close(buys, len(missing))
    if missing:
        print(f"\nincomplete: {len(missing)} ticker(s) absent today: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
