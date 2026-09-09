r"""Earnings dates and post-earnings announcement drift (PEAD).

Two things live here: a fetch/cache layer for announcement dates, and a `pead`
signal following the `signals.py` contract.

## The data turned out to be much better than feared

`yf.Ticker(t).get_earnings_dates(limit=100)` returns up to 100 rows per ticker
reaching back to ~2002 -- not the two years the task budgeted for. For all 100
tickers (the 40 plus the 60-name validation universe) it returned **9638 raw
rows, zero failures**, of which **4396 are in the 2015-05-28 -> 2026-07-17 price
window**: ~44 events per ticker, 396-400 per calendar year, no gaps. Every row
carries `EPS Estimate`, `Reported EPS` and `Surprise(%)`, all non-null in
window, so a real analyst-estimate surprise is available and the abnormal-return
proxy can be cross-checked against it.

These are **true announcement dates**, not period-end dates from
`quarterly_income_stmt` (that endpoint returns only 5 quarters and is useless
here). The timestamp carries the time of day, which recovers the BMO/AMC
convention, and the price data confirms it decisively:

| convention | n | \|r\|/66d mean at t | at t+1 | volume at t | at t+1 |
|---|---|---|---|---|---|
| hour >= 16 (after close) | 530 | 1.10 | **3.89** | 1.57 | **2.72** |
| hour < 10 (before open)  | 1144 | **2.96** | 1.33 | **2.08** | 1.40 |

So the reaction day is t+1 for after-close announcements and t for before-open
ones, and `reaction_day` implements exactly that. Getting this wrong would put a
third of the events one day off and smear the event day into the drift window.

## PEAD: REFUTED as a tradeable rule on this universe

Pre-specified primary, written down before any return was computed (top quintile
of standardized earnings-day abnormal return, side long, entry_lag=1, **h=20**,
10bps, judged on `excess_ret` through `events.full_report`, bar = t_block > 2 and
placebo p < 0.05). Usable events on the 40 after the 252d beta warm-up:
**n = 1570, 2016-09-21 -> 2026-07-16**.

It fails, and not narrowly:

| primary, top quintile z, h=20 | |
|---|---|
| n | 310 (~31/yr) |
| ev_excess | **-0.29%** |
| t_block (40 quarterly blocks) | **-0.72**, p = 0.47 |
| block 95% CI | **[-1.09%, +0.48%]** |
| placebo, 1000 reps | **p = 0.79** |

Wrong sign, no significance, and random entries on the same names beat the
signal in 79% of draws. The useful number is the interval: **any 20-day drift
above +0.48% is ruled out at 95%.** This is a powered negative -- block SE is
0.40%, so a true +0.8% drift would have been seen.

Three things make it a clean negative rather than a shrug:

- The **surprise measure works.** Top-quintile events move 5.74x their normal
  daily range on the reaction day (all events 3.28x), and z correlates with the
  analyst EPS surprise at Spearman +0.25. The information is being measured.
- The **quintile ordering is the wrong shape**, at every horizon -- an inverted
  U, not PEAD's monotone ramp. Q3 (the no-surprise quarters) is the best group
  and both tails lose: at h=20, Q1..Q5 = -0.40% / +0.19% / **+0.33%** / -0.24% /
  -0.29%. One lucky cell can be selection; a whole ordering with the wrong shape
  cannot.
- An **independent surprise proxy agrees.** Re-running the identical rule off
  the analyst EPS surprise gives a flat wash, max |t_block| = 1.39 over eight
  cells. So the abnormal-return proxy is not the culprit.

Controls, both pre-specified:

- **It is not earnings, it is just a big move.** 36% of `signals.idio_shock`
  fires (303 of 848) land within +-3 days of an earnings reaction date. Split
  them and the *non-earnings* shocks are at least as negative as the earnings
  ones at h=10 (-0.65% vs -0.51%) and h=20 (-0.62% vs -0.83%). No
  earnings-specific component survives. Note the corollary: `idio_shock` is
  substantially an earnings signal that was never labelled as one.
- **It is not prior momentum.** Fired names carry mom63 of +2.78% against a
  universe mean of +3.16% -- slightly below average -- and EV is negative in 3 of
  4 momentum quartiles.

Higher-power version, since a quintile cut discards most of the sample:
Spearman rank correlation of surprise against forward excess return over all
1530 events, quarterly block bootstrap. This is the only thread in the study:
rho = +0.033 (t +1.20) at h=10 and +0.032 (t +1.38) at h=20, PEAD's sign in 6 of
8 cells across both surprise measures. Nothing reaches t = 1.4, the h=20 CI is
[-0.013, +0.079], and `signals.reversal_in_trend` already established in this
repo that an IC of +0.02 does not convert into a trade rule. Here it
demonstrably does not -- the tail trade is negative. Do not pull this thread.

Multiple comparisons: the primary is one pre-registered cell and it failed, so
selection is not a defence anyone needs. Across ~56 context cells, two came in
under p = 0.10 and none under 0.05 -- fewer than chance would produce. There is
no hidden result in the grid.

**The 60-ticker validation universe was deliberately not touched.** Confirmation
was conditional on the primary passing on the 40; it did not, so the 60 remains
unspent.

Why this is unsurprising rather than a data failure: PEAD was documented on a
wide cross-section including small, thinly-covered, illiquid names, and the
published effect has decayed substantially since the 1990s. 40 mega-caps with
dozens of analysts each and machine-read filings are the least likely place for
market underreaction to persist. The effect may well still be real elsewhere; it
is not harvestable here.

The durable output of this work is therefore the **data**, not the signal: 4396
verified announcement dates with EPS estimates and actuals across all 100
tickers, reusable as an event-window veto (don't hold into a print), as a
volatility feature, or for any future event study.

`pead` is kept for the same reason `idio_shock` is: it is a correct
implementation of a clean hypothesis that the data declined to support.
"""

import os
import warnings

import numpy as np
import pandas as pd

from alpha import market_return
from signals import dedupe_fires

EARNINGS_DIR = ('/Users/gergelyfazekas/Documents/python_projects/'
                'trading_new_structure/data/earnings')
RAW_FILE = os.path.join(EARNINGS_DIR, 'raw_earnings_dates.csv')


# ---------------------------------------------------------------- fetch/cache

def fetch_earnings_dates(tickers, limit=100, pause=0.6, retries=3):
    """Pull announcement dates from Yahoo. Yahoo hard-caps `limit` at 100.

    Returns raw rows including the tz-aware timestamp, whose *hour* is what
    tells us whether the number hit before the open or after the close.
    """
    import time
    import yfinance as yf

    rows, fails = [], []
    for tkr in tickers:
        for attempt in range(retries):
            try:
                df = yf.Ticker(tkr).get_earnings_dates(limit=limit)
                if df is None or not len(df):
                    raise ValueError('empty')
                d = df.reset_index()
                d.columns = ['ts', 'eps_est', 'eps_act', 'surprise_pct']
                d['stock'] = tkr
                rows.append(d)
                break
            except Exception as exc:                     # noqa: BLE001
                if attempt == retries - 1:
                    fails.append((tkr, repr(exc)[:120]))
                time.sleep(2 + 3 * attempt)
        time.sleep(pause)
    if fails:
        warnings.warn(f'earnings fetch failed for {fails}')
    if not rows:
        return pd.DataFrame(columns=['ts', 'eps_est', 'eps_act', 'surprise_pct', 'stock'])
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(EARNINGS_DIR, exist_ok=True)
    out.to_csv(RAW_FILE, index=False)
    return out


def load_earnings(tickers=None, path=RAW_FILE):
    """Cached announcement dates -> [stock, date, hour, eps_est, eps_act, surprise_pct].

    `date` is the announced calendar date and `hour` its local hour; the pair of
    them is all `reaction_day` needs. Duplicate (stock, date) rows -- Yahoo emits
    a few for pre-2010 quarters -- are dropped.
    """
    raw = pd.read_csv(path)
    ts = [pd.Timestamp(s) for s in raw.ts]
    out = pd.DataFrame({
        'stock': raw.stock.values,
        'date': pd.to_datetime([t.date() for t in ts]),
        'hour': [t.hour for t in ts],
        'eps_est': raw.eps_est.values,
        'eps_act': raw.eps_act.values,
        'surprise_pct': raw.surprise_pct.values,
    })
    if tickers is not None:
        out = out[out.stock.isin(list(tickers))]
    return (out.drop_duplicates(['stock', 'date'])
               .sort_values(['stock', 'date'])
               .reset_index(drop=True))


def reaction_day(index, date, hour, amc_hour=16):
    """Map an announced (date, hour) to the trading day the market reacts on.

    hour >= amc_hour  -> announced after the close -> the reaction is the *next*
    session. Otherwise the number was out before or during the session, so the
    reaction is that session. Returns a positional index into `index`, or None.

    Verified against the data, not assumed: after-close events show 3.9x normal
    |return| and 2.7x normal volume on t+1 and nothing on t; before-open events
    show 3.0x and 2.1x on t. See the module docstring.
    """
    p = index.searchsorted(pd.Timestamp(date))
    if p >= len(index):
        return None
    if hour >= amc_hour:
        p += 1
    if p >= len(index):
        return None
    return int(p)


# ------------------------------------------------------------------- surprise

def earnings_surprise(stock_list, earn=None, beta_window=252, resid_vol_window=66,
                      return_col='log_return', amc_hour=16):
    """Attach the standardized earnings-day abnormal return to every event.

    Identical construction to `signals.idio_shock`, which is the point -- it
    makes the two directly comparable, so `idio_shock` fires on non-earnings
    days are a like-for-like control for "is this just a shock, or is it an
    *earnings* shock":

        resid = r_i - beta_i * r_m       beta over `beta_window` vs equal weight
        z     = resid / resid.rolling(resid_vol_window).std().shift(1)

    The `.shift(1)` keeps the event day out of its own denominator. No market
    veto here (unlike `idio_shock`): an earnings release is an event whatever
    the index did that day.

    Returns the event frame plus `react_date`, `z`, `resid`, `abs_ret_norm`.
    """
    if earn is None:
        earn = load_earnings(stock_list.tickers)
    mkt = market_return(stock_list, return_col=return_col)

    rows = []
    for tkr in stock_list.tickers:
        ev = earn[earn.stock == tkr]
        if not len(ev):
            continue
        df = stock_list[tkr].data
        idx = pd.to_datetime(df.index)
        r = pd.Series(df[return_col].values, index=idx).astype(float)
        m = mkt.reindex(idx)

        beta = r.rolling(beta_window).cov(m) / m.rolling(beta_window).var().replace(0, np.nan)
        resid = r - beta * m
        sigma = resid.rolling(resid_vol_window).std().shift(1)
        z = resid / sigma.replace(0, np.nan)
        absn = r.abs() / r.abs().rolling(resid_vol_window).mean().shift(1)

        for rec in ev.itertuples():
            p = reaction_day(idx, rec.date, rec.hour, amc_hour=amc_hour)
            if p is None:
                continue
            rows.append({'stock': tkr, 'ann_date': rec.date, 'hour': rec.hour,
                         'react_date': idx[p], 'z': z.iloc[p], 'resid': resid.iloc[p],
                         'abs_ret_norm': absn.iloc[p],
                         'eps_est': rec.eps_est, 'eps_act': rec.eps_act,
                         'surprise_pct': rec.surprise_pct})
    if not rows:
        return pd.DataFrame(columns=['stock', 'ann_date', 'react_date', 'z'])
    return pd.DataFrame(rows).dropna(subset=['z']).reset_index(drop=True)


# --------------------------------------------------------------------- signal

def pead(stock_list, q=0.8, arm='up', surprise_col='z', events=None,
         min_gap=10, side='long', **kw):
    """Fire on earnings events in the extreme tail of the earnings-day surprise.

    The hypothesis is the textbook one: a stock that jumps on its earnings keeps
    drifting the same way for weeks, because the market underreacts to the
    information in the release. Surprise is proxied by the **standardized
    earnings-day abnormal return** (see `earnings_surprise`), the standard stand-in
    when analyst estimates are unavailable -- though here they *are* available, so
    `surprise_col='surprise_pct'` runs the same rule off the analyst-estimate
    surprise as an independent check.

    `q` is a quantile of the pooled surprise distribution: `q=0.8, arm='up'`
    fires on the top quintile, `arm='down'` on the bottom `1-q`.

    **REFUTED -- do not trade this.** The pre-specified primary was the top
    quintile at h=20 on the 40, and it does not clear its bar. Read the module
    docstring and `scratchpad/report_earnings.md` before reusing any of it. The
    negative is a clean one: the event dates are verified true announcement
    dates, the event day is unmistakable in return and volume, ~1760 in-sample
    events is a respectable sample, and the drift still is not there. Nor is the
    quintile ordering monotone, which is the check a single lucky cell cannot
    fake.

    The down arm is reported as context only. This is a long-only book, so the
    actionable version of a negative view is declining the name, not shorting it.
    """
    if events is None:
        events = earnings_surprise(stock_list, **kw)
    x = events[surprise_col].astype(float)
    thr = x.quantile(q if arm == 'up' else 1 - q)
    hit = events[(x >= thr) if arm == 'up' else (x <= thr)]

    fires = pd.DataFrame({'stock': hit.stock.values,
                          'fire_date': hit.react_date.values,
                          'side': side,
                          'strength': hit[surprise_col].astype(float).values})
    if not len(fires):
        return pd.DataFrame(columns=['stock', 'fire_date', 'side', 'strength'])
    return dedupe_fires(fires, min_gap=min_gap)
