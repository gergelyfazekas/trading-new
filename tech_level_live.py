"""Daily recompute of live technical levels for the hand-traded ticker list.

Reuses tech_levels.py's causal construction directly against freshly-pulled
price data -- no bronze/silver/gold tiers needed, this only ever needs
close/volume -- to report the closest known level below and above the current
price, for hand-trading decisions.

Two arms are reported per ticker, deliberately:

  fixed  -- one combo, identical for every ticker, chosen a priori from the
            universe-wide calibration in tech_level_search.DEFAULT_GRID
            (prominence 0.01 = universe p50; distance 10 = grid middle;
            tech_width 0.008 = middle of the tight 0.003-0.012 range that is
            structurally immune to the width-inflation artifact). Nothing about
            it was selected on any individual stock's outcome, so the forward
            record it generates carries no selection bias.
  tuned  -- that ticker's own grid-search winner, the way AAPL's combo was
            picked. Stronger if levels genuinely need per-name parameters,
            but selected out of 84 combos per stock, which is exactly the
            multiple-comparisons exposure tech_levels_notes.md flags.

The fixed arm is the one to trade off. The tuned arm is shown so you can see
whether per-name tuning would have mattered -- not so you can take whichever
arm happens to show a level nearby on a given morning. The script prints an
explicit agreement line to keep that choice visible rather than rationalisable.

Only AAPL and AMZN carry a tuned arm, and no further ones should be added. The
universe was widened to config.ticker_list (40 names) on 2026-09-03 to raise the
event count; grid-searching a tuned combo for each would have meant 40 x 84
selections on the same forward record, which is the multiple-comparisons
exposure tech_levels_notes.md flags, at 20x the previous scale. The 38 new names
run fixed-only, which makes their record cleaner than AAPL's and AMZN's, not
worse. config.oos_ticker_list was deliberately NOT used: those 60 names are
reserved for historical out-of-sample confirmation and logging them forward here
would spend them for no gain, since this record is already out-of-sample in time.

Only the level BELOW is a long-only actionable setup: price falling into a band
from above is a support test, which is the direction every EV result in
tech_levels_notes.md was measured on. The level above is a resistance test --
shown for context, but an equal-weight book can only express it by trimming.

Run daily: ./venv/bin/python tech_level_live.py
"""
import json
import datetime
import os
import socket
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

from stock_class import StockList
from tech_levels import find_touches, build_levels_causal

COMBO_FILE = os.path.join(os.path.dirname(__file__), "data", "live_combos.json")
LOG_FILE = os.path.join(os.path.dirname(__file__), "data", "live_log.csv")
START_DATE = datetime.date(2015, 5, 28)

# Batched download retry. The failure this guards against is yfinance returning
# an empty frame ("possibly delisted; no price data found") on a transient API
# error or rate-limit -- 70 times between 2026-08-13 and 2026-08-24, which is
# what cost the record 7 of its first 20 trading days.
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_BACKOFF = 30      # seconds, doubled each retry
SOCKET_TIMEOUT = 60        # cap a single stalled read; see run_live_log.sh

EXIT_UNSETTLED = 3         # bar still forming -- abort, do NOT retry

LOG_COLUMNS = [
    'as_of', 'ticker', 'arm', 'price', 'inside',
    'inside_low', 'inside_high', 'inside_birth', 'inside_age',
    'support_low', 'support_high', 'support_birth', 'support_age', 'support_pct',
    'resist_low', 'resist_high', 'resist_birth', 'resist_age', 'resist_pct',
    'n_levels', 'combo_distance', 'combo_prominence', 'combo_tech_width',
    'bar_settled', 'run_date',
]


def load_combos(path=COMBO_FILE):
    """Return (fixed_combo, {ticker: {arm: combo}}).

    Also accepts the pre-2026-08 flat {ticker: combo} layout, treating each
    entry as that ticker's tuned combo with no fixed arm, so an old file still
    reports something rather than raising.
    """
    with open(path) as f:
        raw = json.load(f)

    if 'tickers' not in raw:
        return None, {t: {'tuned': c} for t, c in raw.items()}
    return raw.get('fixed'), raw['tickers']


def save_combo(ticker, combo, arm='tuned', path=COMBO_FILE):
    """Persist a per-ticker combo for the daily script to pick up.

    arm='tuned' (the default) sets that ticker's own grid-search winner. The
    fixed arm is shared across every ticker and is set with set_fixed_combo.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
        if 'tickers' not in raw:  # migrate the old flat layout in place
            raw = {'fixed': None, 'tickers': {t: {'tuned': c} for t, c in raw.items()}}
    except FileNotFoundError:
        raw = {'fixed': None, 'tickers': {}}

    raw['tickers'].setdefault(ticker, {})[arm] = combo
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def set_fixed_combo(combo, path=COMBO_FILE):
    """Replace the shared a-priori combo applied to every ticker.

    Changing this resets the forward record: events logged before and after are
    no longer the same experiment. Note the change in tech_levels_notes.md.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
        if 'tickers' not in raw:
            raw = {'fixed': None, 'tickers': {t: {'tuned': c} for t, c in raw.items()}}
    except FileNotFoundError:
        raw = {'fixed': None, 'tickers': {}}

    raw['fixed'] = combo
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)


def current_levels(ticker, combo, close=None, volume=None):
    """Build today's set of levels for one (ticker, combo).

    close/volume: already-pulled series. Passing them lets both arms share one
    download per ticker -- the arms differ only in peak-detection parameters,
    never in the underlying prices.
    """
    if close is None:
        close, volume = pull_series(ticker)

    combo = dict(combo)
    tech_width = combo.pop('tech_width')
    consider_volume = combo.pop('consider_volume', False)
    volume_height = combo.pop('volume_height', 0.0)
    volume_prominence = combo.pop('volume_prominence', 0.0)

    touches = find_touches(close, volume=volume, consider_volume=consider_volume,
                            volume_height=volume_height, volume_prominence=volume_prominence,
                            **combo)
    levels = build_levels_causal(touches, tech_width)
    return levels, float(close.iloc[-1]), close.index[-1]


def pull_all(tickers, attempts=DOWNLOAD_ATTEMPTS, backoff=DOWNLOAD_BACKOFF, include_hl=False):
    """Fresh full-history (close, volume) for every ticker in ONE download,
    or (close, high, low, volume) when include_hl=True.

    Returns ({ticker: (close, volume) or (close, high, low, volume)},
    [tickers that came back with no data]). include_hl defaults to False so
    every existing caller (which unpacks a 2-tuple) is unaffected;
    yfinance's download already carries high/low, this just chooses whether
    to surface them.

    Batching matters more than it looks. The previous shape called
    StockList([ticker]) once per name, so a 40-ticker run meant 40 separate
    yfinance calls -- 40 independent chances to hit the rate limit, and any one
    of them raised out of main() before append_log ever ran, losing the whole
    day including the tickers that had already succeeded. That is exactly how
    2026-08-13 through 2026-08-21 went missing with only two tickers; at 40 it
    would have lost nearly every day. One batched call plus per-ticker
    isolation below turns a total loss into, at worst, a few absent names.
    """
    end = datetime.date.today() + datetime.timedelta(days=1)
    tickers = list(tickers)
    last_err = None

    for attempt in range(1, attempts + 1):
        try:
            s = StockList(tickers)
            s.pull_data(start=START_DATE, end=end)
        except Exception as exc:
            # An entirely empty download leaves stock_class.set_dates indexing
            # an empty frame -> IndexError. Treat any pull failure as retryable.
            last_err = exc
            if attempt < attempts:
                wait = backoff * (2 ** (attempt - 1))
                print(f"  download attempt {attempt}/{attempts} failed ({exc.__class__.__name__}: {exc}); "
                      f"retrying in {wait}s")
                time.sleep(wait)
            continue

        series, failed = {}, []
        for ticker in tickers:
            try:
                data = s[ticker].data
                close = data['close'].dropna()
                if close.empty:
                    failed.append(ticker)
                    continue
                volume = data['volume'].reindex(close.index) if 'volume' in data.columns else None
                if include_hl:
                    high = data['high'].reindex(close.index) if 'high' in data.columns else None
                    low = data['low'].reindex(close.index) if 'low' in data.columns else None
                    series[ticker] = (close, high, low, volume)
                else:
                    series[ticker] = (close, volume)
            except Exception:
                failed.append(ticker)

        if series:
            return series, failed
        last_err = RuntimeError("every ticker came back empty")
        if attempt < attempts:
            wait = backoff * (2 ** (attempt - 1))
            print(f"  download attempt {attempt}/{attempts}: no ticker returned data; retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"download failed after {attempts} attempts: {last_err}")


def pull_series(ticker):
    """Fresh full-history (close, volume) for one ticker. Thin wrapper on pull_all."""
    series, failed = pull_all([ticker])
    if ticker in failed or ticker not in series:
        raise RuntimeError(f"no price data for {ticker}")
    return series[ticker]


def bar_is_settled(as_of, now=None):
    """Has the US session dated `as_of` actually closed?

    yfinance serves the current, still-forming bar during market hours, so a run
    that drifts past 09:30 ET records an intraday print as if it were the close.
    That is not hypothetical: on 2026-08-14 the shell retry loop reached attempt
    3 at 15:38 CEST (09:38 ET) and logged as_of=2026-08-14 eight minutes into the
    session. Those four rows carry bar_settled=False.

    Compared against US Eastern rather than local time so it stays correct
    through both DST transitions, which fall on different dates in the EU and US.
    """
    et = ZoneInfo("America/New_York")
    now = now or datetime.datetime.now(et)
    as_of = pd.Timestamp(as_of).date()
    if as_of != now.date():
        return as_of < now.date()
    return now.time() >= datetime.time(16, 15)


def _sessions_since(bar_index, birth_date, as_of):
    """Trading sessions between a level's birth and this reading.

    Counted off the actual bar index, not a business-day range, so exchange
    holidays don't inflate it. Sessions is the unit that matters because it is
    the unit `distance` is expressed in: a level younger than `distance`
    sessions sits in the provisional zone where find_peaks can still un-detect
    its confirming touch as later bars arrive, so it may be revised away
    entirely -- see append_log's note. Compare age against combo_distance to
    flag those rows; provisional_mask() does it.
    """
    if birth_date is None or (isinstance(birth_date, float) and pd.isna(birth_date)):
        return None
    birth = pd.Timestamp(birth_date)
    end = pd.Timestamp(as_of)
    return int(((bar_index > birth) & (bar_index <= end)).sum())


def provisional_mask(df, side):
    """Rows where `side` ('support'/'resist'/'inside') was still revisable.

    A level whose age has not yet exceeded the run's `distance` parameter can
    still be un-born by a later bar. AAPL's [314.93, 316.83] band is the worked
    example: shown by the fixed arm for five sessions from 2026-08-21, then gone
    on 2026-08-31 with n_levels dropping 105 -> 104. Segment on this before
    pooling; do not silently mix revisable and confirmed levels.
    """
    return df[f'{side}_age'].notna() & (df[f'{side}_age'] <= df['combo_distance'])


def _neighbours(levels, current_price):
    """(containing, closest_below, closest_above) for the current price."""
    containing = [lvl for lvl in levels if lvl.band[0] <= current_price <= lvl.band[1]]
    below = [lvl for lvl in levels if lvl.band[1] < current_price]
    above = [lvl for lvl in levels if lvl.band[0] > current_price]
    return (
        containing,
        max(below, key=lambda lvl: lvl.band[1]) if below else None,
        min(above, key=lambda lvl: lvl.band[0]) if above else None,
    )


def _describe(arm, levels, current_price):
    """Lines for one arm, plus whether price is currently inside a band."""
    containing, closest_below, closest_above = _neighbours(levels, current_price)
    lines = []
    for lvl in containing:
        lines.append(f"  [{arm:5}] ** INSIDE a known level: "
                     f"[{lvl.band[0]:.2f}, {lvl.band[1]:.2f}] (born {lvl.birth_date})")
    if closest_below:
        dist = (current_price - closest_below.band[1]) / current_price
        lines.append(f"  [{arm:5}] support below:  "
                     f"[{closest_below.band[0]:.2f}, {closest_below.band[1]:.2f}] "
                     f"(born {closest_below.birth_date}, {dist:.2%} below)")
    else:
        lines.append(f"  [{arm:5}] support below:  none known")
    if closest_above:
        dist = (closest_above.band[0] - current_price) / current_price
        lines.append(f"  [{arm:5}] resistance above: "
                     f"[{closest_above.band[0]:.2f}, {closest_above.band[1]:.2f}] "
                     f"(born {closest_above.birth_date}, {dist:.2%} above)")
    else:
        lines.append(f"  [{arm:5}] resistance above: none known")
    return lines, bool(containing)


def _record(ticker, arm, combo, levels, current_price, as_of, bar_index, settled=True):
    """One log row: what this arm saw for this ticker on this date.

    bar_index: the ticker's DatetimeIndex of bars, used to age each reported
    level in trading sessions. Age is what separates a level with months of
    confirmation from one born last week and still revisable -- the single
    distinction the first 13 days of this log could not make.
    """
    containing, below, above = _neighbours(levels, current_price)
    inner = containing[0] if containing else None
    return {
        'as_of': pd.Timestamp(as_of).date(),
        'ticker': ticker,
        'arm': arm,
        'price': round(current_price, 4),
        'inside': bool(containing),
        'inside_low': round(inner.band[0], 4) if inner else None,
        'inside_high': round(inner.band[1], 4) if inner else None,
        'inside_birth': pd.Timestamp(inner.birth_date).date() if inner else None,
        'inside_age': _sessions_since(bar_index, inner.birth_date, as_of) if inner else None,
        'support_low': round(below.band[0], 4) if below else None,
        'support_high': round(below.band[1], 4) if below else None,
        'support_birth': pd.Timestamp(below.birth_date).date() if below else None,
        'support_age': _sessions_since(bar_index, below.birth_date, as_of) if below else None,
        'support_pct': round((current_price - below.band[1]) / current_price, 6) if below else None,
        'resist_low': round(above.band[0], 4) if above else None,
        'resist_high': round(above.band[1], 4) if above else None,
        'resist_birth': pd.Timestamp(above.birth_date).date() if above else None,
        'resist_age': _sessions_since(bar_index, above.birth_date, as_of) if above else None,
        'resist_pct': round((above.band[0] - current_price) / current_price, 6) if above else None,
        'n_levels': len(levels),
        # carried per row so a later set_fixed_combo / save_combo is visible in
        # the log itself -- segment on these before pooling anything.
        'combo_distance': combo.get('distance'),
        'combo_prominence': combo.get('prominence'),
        'combo_tech_width': combo.get('tech_width'),
        # False marks a row taken off a still-forming bar. main() refuses to
        # write those now; the column exists because four already exist.
        'bar_settled': bool(settled),
        'run_date': datetime.date.today(),
    }


def append_log(records, path=LOG_FILE):
    """Upsert today's rows into the forward record, keyed on (as_of, ticker, arm).

    THIS LOG MUST NEVER BE REGENERATED FROM HISTORY. Its entire value is that
    each row is what the detector actually showed *on that date* -- and the
    newest ~`distance` days of levels are provisional (a local extremum can't be
    confirmed until price moves away from it, see the prominence-scaling section
    of tech_levels_notes.md), so recomputing an old date yields a level set the
    decision could not have used. Backfilling would silently convert an
    out-of-sample record into an in-sample one.

    Re-running on the same day overwrites that day's rows rather than appending
    duplicates, so a second run after a data revision is safe; running on a
    non-trading day just rewrites the last trading day's rows unchanged.
    """
    if not records:
        return None

    new = pd.DataFrame(records, columns=LOG_COLUMNS)
    if os.path.exists(path):
        old = pd.read_csv(path)
        # match dtypes so the anti-join below compares like with like
        old['as_of'] = pd.to_datetime(old['as_of']).dt.date
        keys = set(zip(new['as_of'], new['ticker'], new['arm']))
        mask = [k not in keys for k in zip(old['as_of'], old['ticker'], old['arm'])]
        combined = pd.concat([old[mask], new], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        combined = new

    # Tolerate a schema that has grown since the file was written: rows logged
    # under an older LOG_COLUMNS keep their values and get NaN for the new
    # fields. Those blanks are correct and must stay blank -- a column added
    # today cannot be filled in for a past date without recomputing that date,
    # which is the one thing the note above forbids. (The one-off ages/settled
    # backfill was a separate, documented exception: it derived only from
    # birth_date and as_of, both already in the row.)
    combined = combined.reindex(columns=LOG_COLUMNS)
    combined = combined.sort_values(['as_of', 'ticker', 'arm']).reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def report(ticker, arms, close=None, volume=None, settled=True):
    """Print every configured arm for one ticker off a single price series.

    close/volume: already pulled by main()'s batched download. Left optional so
    the function is still usable interactively for a single name.

    Returns the log rows for this ticker.
    """
    if close is None:
        close, volume = pull_series(ticker)
    bar_index = pd.DatetimeIndex(pd.to_datetime(close.index))

    printed_header = False
    inside = {}
    records = []
    for arm, combo in arms.items():
        if combo is None:
            continue
        levels, current_price, as_of = current_levels(ticker, combo, close=close, volume=volume)
        if not printed_header:
            print(f"{ticker} as of {as_of}  (current price {current_price:.2f})")
            printed_header = True
        lines, is_inside = _describe(arm, levels, current_price)
        inside[arm] = is_inside
        records.append(_record(ticker, arm, combo, levels, current_price, as_of,
                               bar_index, settled=settled))
        for line in lines:
            print(line)

    # Make the two arms' verdicts explicit. The reason for showing both is to
    # learn whether per-name tuning matters -- not to pick whichever arm is
    # convenient today, so a disagreement is called out rather than left to be
    # noticed (or not).
    if len(inside) == 2:
        if all(inside.values()):
            print("  -> both arms agree: price is at a level")
        elif not any(inside.values()):
            print("  -> both arms agree: price is not at a level")
        else:
            at = [a for a, v in inside.items() if v][0]
            print(f"  -> ARMS DISAGREE: only '{at}' puts price at a level. "
                  f"Trade the fixed arm.")
    print()
    return records


def main():
    socket.setdefaulttimeout(SOCKET_TIMEOUT)

    fixed, tickers = load_combos()
    if fixed is None:
        print("warning: no fixed combo set; reporting tuned arm only\n")

    series, failed = pull_all(tickers.keys())
    if failed:
        print(f"warning: no data for {len(failed)} ticker(s): {', '.join(sorted(failed))}\n")

    records, unsettled, errored = [], [], []
    for ticker, arms in tickers.items():
        if ticker not in series:
            continue
        close, volume = series[ticker]

        # Never write a still-forming bar. A missing day is a gap; a partial bar
        # is a wrong number that looks exactly like a right one.
        as_of = close.index[-1]
        if not bar_is_settled(as_of):
            unsettled.append(ticker)
            continue

        # One bad ticker must not cost the other 39 their row.
        try:
            records.extend(report(ticker, {'fixed': fixed, 'tuned': arms.get('tuned')},
                                  close=close, volume=volume))
        except Exception as exc:
            errored.append(ticker)
            print(f"{ticker}: SKIPPED -- {exc.__class__.__name__}: {exc}\n")

    if unsettled:
        print(f"ABORTED: latest bar for {len(unsettled)} ticker(s) has not settled "
              f"({', '.join(sorted(unsettled)[:5])}...). Nothing logged -- rerun after "
              f"16:15 US/Eastern, or let tomorrow's 08:00 run pick it up.")
        return EXIT_UNSETTLED

    combined = append_log(records)
    if combined is not None:
        at_level = combined[combined['inside'].astype(bool)]
        print(f"logged {len(records)} rows to {os.path.relpath(LOG_FILE, os.path.dirname(__file__))} "
              f"({len(combined)} total, {combined['as_of'].nunique()} trading days, "
              f"{combined['ticker'].nunique()} tickers, {len(at_level)} at-level events so far)")

    missing = sorted(set(failed) | set(errored))
    if missing:
        # Non-zero so run_live_log.sh retries and fills the absent names in via
        # the (as_of, ticker, arm) upsert. A name that fails every attempt every
        # day is a dead ticker, not a blip -- take it out of live_combos.json.
        print(f"\nincomplete: {len(missing)} ticker(s) absent from today's rows: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
