# Short-term mean reversion — candidate signals (2026-09-22)

**Status.** Nothing tested yet. This file exists to hold candidates before any
backtest, so the design constraint below is applied from the start instead of
retrofitted.

## Why this file exists

The Five-Day Bounce (two-touch technical level) strategy was disproved on
2026-09-21 by a fully causal day-by-day test: whole-history backtest showed
+0.91% excess/trade (t=+7.5), day-by-day (levels rebuilt from `close[:i]`
only, at every day `i`) showed **-0.20%** (t=-1.06). Every variant tried on
top of it also failed causally: bare local low, same-type pairs, gap cap,
confirmation delay, tight troughs, double-bottom/retest. See
`tech_levels_notes.md`, "Day-by-day (fully causal) test of the frozen rule"
(2026-09-21), and `strategies/five_day_bounce/NOTES.md`.

**Root cause, and the constraint it implies.** All of it used
`find_peaks(distance=N)`-style extremum detection: a bar only counts as a
trough if no lower low appears in the *following* N bars. That's future
information leaking into a same-day signal. Splitting day-by-day trades by
whether hindsight would have kept them: hindsight-kept trades show +0.81%
excess, the ones hindsight discards ("false starts") show **-0.92%**
(t=-3.69) — the whole-history edge was an artifact of silently averaging
over the confirmed survivors only.

**Design rule for everything below:** no signal may be defined by a
condition that needs bars after the fire date to evaluate. "Local low"
must mean a backward-only rolling extremum (`rolling_min`,
`argmin` over a trailing window, current-bar drawdown from a trailing
rolling max) — never a `scipy.find_peaks`-style comparison that looks past
the fire date. Any candidate here should be testable directly in the
`events.py` harness (`entry_lag=1`, block-bootstrap by quarter, placebo
built in) or via the day-by-day truncate-and-rebuild pattern in
`strategies/five_day_bounce/experiments/original_rule_daybyday.py` if it
still needs a rolling-window recompute per day.

The user still believes a real short-term mean-reversion effect exists;
the task is finding the correctly (causally) specified version of it, not
abandoning the idea.

## Candidates

### Reformed local-trough family
Keeps the original "buy near a recent low" premise, redefines "low" so it
never needs future bars.

1. **Rolling N-day low, no confirmation wait.** `close[i] == rolling_min(close, N)[i]`
   — a new N-day low, known the instant it happens. The direct causal
   analogue of the old rule; N (10/20/60) replaces the old `distance`
   parameter. Most literal repair of the exact thing that just failed —
   good first candidate.
2. **Trailing drawdown-depth z-score.** `(close - rolling_max(close, N)) / rolling_std`
   — z-scores distance below the recent high, normalized so "deep dip"
   means the same thing across tickers/regimes instead of a fixed %.
3. **Streak-based reversal.** Length or cumulative decline of the current
   consecutive-down-day streak, z-scored against that ticker's trailing
   streak-length distribution. Reframes "just made a low" as "has fallen
   longer/harder than usual."
4. **New-low + capitulation volume.** #1 combined with a same-bar volume
   z-score. Revisits the step-19 volume-gating idea, but both the low and
   the volume spike are defined causally at bar close, so there's no
   confirmation lag to leak through.

### General trailing z-score reversal family
Statistical extremity, no reference to "a low" specifically — useful both
as independent candidates and as placebos against the family above.

5. **Return z-score (classic short-term reversal).** z-score of trailing
   3-5 day cumulative return vs. its own 60-90 day rolling distribution;
   buy the most negative tail. Textbook Jegadeesh/Lehmann effect — likely
   weak/arbed in large caps, but a clean, well-understood baseline.
6. **Distance-from-moving-average z-score.** `(close - SMA_20) / rolling_std_20`
   — a Bollinger %B without the discrete "band touch," so there's no level
   for future info to leak through.
7. **RSI(2)/RSI(3) extreme oversold.** Same family as #6, bounded
   oscillator instead of a raw z-score (Connors-style). Compare against #6
   to see if the nonlinearity matters.
8. **Cross-sectional rank reversal.** Within-date rank of return vs.
   sector/peer group, buy the worst-ranked. Reverts against peers, not own
   history — a different axis, useful as an independent-mechanism check
   alongside the others.

## Two-touch low, fully specified (2026-09-22)

A refinement of candidate #1 (rolling N-day low) into a full two-touch
pattern, kept causal by construction: every forward-looking check is
bounded to at most 2 days ahead, and the buy is dated at whichever day the
pattern actually completes rather than backdated to the low. "Volume"
throughout means the trailing-10 standardized ratio from
`tech_level_walkforward_demo.py`'s `trailing_avg_volume` /
`volume_ratio` (`volume[j] / mean(volume[j-10:j])`, strictly excluding day
j) -- never raw volume.

**Stage 1 -- first low at day L, confirmed as of day L+1's close.**
Looking backward only:
- `close[L] < close[L-1], ..., close[L-lookback]` -- a backward-only
  rolling local minimum over `lookback` trading days. This is the direct
  causal replacement for `find_peaks(distance=N)`.
- `close[L-1] >= close[L] * (1 + prominence_pct)` -- prominence checked
  against day L-1 only, not the whole `lookback` window.
- `volume_ratio[L] > volume_threshold` -- capitulation-volume gate on the
  low day itself.

**Stage 2 -- buy trigger, branching on the size of the L+1 rebound.**
Let `pct1 = (close[L+1] - close[L]) / close[L]`; the branch requires
`pct1 > 0` (L+1 must close above L at all).
- **Flat branch:** `0 < pct1 <= k_pct` -> buy fires at L+1's close.
  L+1's volume is not checked -- fires regardless.
- **Bigger-rebound branch:** `k_pct < pct1 <= rebound_max_pct` **and**
  `volume_ratio[L+1] < rebound_volume_threshold` (weak volume on the
  rebound -- a ceiling, not a floor, the one variable that runs the
  opposite direction from the others) -> check day L+2: buy fires at
  L+2's close if `close[L] <= close[L+2] <= close[L+1]` (pulled back from
  the L+1 peak, not below the original low, inclusive of both ends)
  **and** `volume_ratio[L+2] > volume_threshold_buy` (renewed buying
  interest on the retest).
- Otherwise (rebound too big, or L+1 volume too high, or L+2 outside the
  `[close[L], close[L+1]]` band, or L+2 volume too low): no buy, setup
  dead.

**Variables (all to be swept):**

| name | gates | direction |
|---|---|---|
| `lookback` | backward window for the first-low check | days, 1-10 |
| `prominence_pct` | `close[L-1]` vs. `close[L]` | floor |
| `volume_threshold` | `volume_ratio[L]` | floor, "> 1" as a starting point |
| `k_pct` | splits flat vs. bigger-rebound branch | -- |
| `rebound_max_pct` | outer cap on `pct1` | ceiling |
| `rebound_volume_threshold` | `volume_ratio[L+1]`, bigger-rebound branch only | **ceiling** (weak volume required) |
| `volume_threshold_buy` | `volume_ratio[L+2]`, bigger-rebound branch only | floor |

**Still inherited, not yet revisited:** exit rule, holding period, and
cost convention (5-day hold, 10bps round-trip, one position per ticker,
from the old frozen rule) -- carry over as defaults unless changed.

**Starting parameter values (locked 2026-09-22).** Chosen loose on purpose
to get enough signals to see the shape of things, then tighten from here:

| variable | value | note |
|---|---|---|
| `lookback` | 5 days | ~half the old `find_peaks(distance=10)`, one-directional only |
| `prominence_pct` | 0.5% | loosened from the frozen rule's 1% |
| `volume_threshold` | 1.0 | baseline from the birth-volume gate experiment |
| `k_pct` | 0.5% | loosened from 0.3% |
| `rebound_max_pct` | 2% | loosened from 1.5% |
| `rebound_volume_threshold` | 1.0 | ceiling, unchanged |
| `volume_threshold_buy` | 0.8 | loosened from 1.0 (floor, now allows some below-average volume) |

Exit/cost convention inherited unchanged from the old frozen rule pending
a reason to change it: 5-day hold, 10bps round-trip, one position per
ticker.

**Reminder for later sweeps:** the birth-volume gate experiment ("Birth-day
volume" section, `tech_levels_notes.md`) found that naive threshold sweeps
on a `volume_ratio`-style variable produce a smooth-looking improvement
curve purely from nested subsets, not a real dose-response (Spearman
rho=0.022, p=0.45 once corrected). When `volume_threshold`,
`rebound_volume_threshold`, or `volume_threshold_buy` get swept, use
equal-sized quantile buckets, not a naive threshold sweep.

**Live (2026-09-22).** The `rebound_retest` branch (retest_gap=2, locked
params) is now running as a forward-record live signal in
`strategies/two_touch_low/` -- see that folder's `NOTES.md` and
`two_touch_low_live.py`'s module docstring for the full status caveat.
Not yet wired to scheduled automation (launchd); the scripts exist,
nothing is scheduled to run them automatically yet.

**Extreme-trade inspection and risk overlays (2026-09-22, same day,
later).** Charting the biggest winners and losers found they're
indistinguishable at entry -- both are quiet, ordinary-looking setups --
and the extremes are driven by news/macro events landing during the
5-day hold (2020 COVID crash, 2023 SVB banking crisis, earnings gaps,
market-wide rallies), not by anything in the pattern itself. Tested three
overlays in response:

- **Earnings-date filter** (skip if the ticker has an earnings reaction
  day anywhere in the hold window, using `earnings.py`'s existing
  cache): net negative to neutral (t=1.80 -> 1.70 on the 535-trade
  population). Correctly removed the earnings-driven losers (APD, AXP)
  but also removed earnings-driven winners (TMO, NFLX, AMZN, LLY) that
  were larger on average -- indiscriminately removing earnings-adjacent
  trades costs more upside than it saves downside.
- **Below-L stop-loss** (exit if price closes below the original low),
  the direct analogue of the five-day-bounce's resistance-based early
  exit: **rejected**. At the exact level it fires on 270/535 trades
  (50%), collapsing hit rate 61.3%->46.4% and t to 0.91 -- far too tight,
  gets whipsawed by ordinary post-entry noise the strategy is supposed to
  tolerate. Loosening to a 1-5% buffer below L only ever converges back
  toward the no-stop baseline as it fires less often; it never beats the
  baseline at any buffer tested, alone or combined with the drawdown exit
  below.
- **Market-wide drawdown exit** (exit if SPY closes >= dd_pct below its
  own close on this trade's entry day, checked fresh each session of the
  hold): **validated, now live**. At dd_pct=1%, cuts the four crash-era
  losses roughly in half to two-thirds (SCHW -32%->-13%, AAPL -15%->-7%,
  CSCO -14%->-5%, JPM -12%->-4%) while leaving APD/AXP's non-market-wide
  losses untouched -- it targets exactly the risk it's designed for and
  nothing else. Reduces the *variance* of returns more than it changes
  the average (excess return ~unchanged), which is what pushes the
  t-stat up.

A **true day-by-day causal implementation** of the drawdown exit
(`two_touch_low_daybyday_dd_exit.py` -- loops one day at a time, every
entry/exit independently re-verified on freshly re-sliced `.iloc[:i+1]`
arrays, same "causality checked by truncation" convention as the rest of
this candidate) found the earlier post-hoc, population-fixed estimate had
been *incomplete*: an early exit frees a ticker's slot sooner, letting
additional trades fire that a fixed-hold population never sees. It also
surfaced and resolved a real discrepancy: the original day-by-day harness
let a `flat` signal block a `rebound_retest` slot even though `flat`
isn't traded, which doesn't match what the live script actually does.
The corrected, live-matching numbers (full 101-ticker universe, no
re-gating games, `flat` never blocks):

| | n | hit | excess | t |
|---|---|---|---|---|
| 5-day hold only | 566 | 62.9% | +0.28% | +2.06 |
| + market-drawdown exit, 1% | 567 | 60.1% | +0.28% | **+2.40** |

Replicated on the 81 tickers never touched while tuning any of this
(params, retest_gap, or dd_pct): 5-day-hold-only t=+1.99, with the
drawdown exit t=+2.13 -- same direction, three consistent confirmations
(in-sample, fresh-81 alone, combined), no reversals anywhere. This
supersedes the originally-published n=535/t=+1.80 figure, which used the
non-live-matching busy-tracking; the 535 number is retired, not
corrected in place, so it isn't silently overwritten in history.
`two_touch_low_live.py` now implements the market-drawdown exit
(`DD_PCT=0.01`) and pulls SPY alongside the 101 tickers each run; tested
working end-to-end 2026-09-22.

A same-day side test of the user's hypothesis ("bigger fall before the
low, on high volume, then a retest -> bigger upside") found only a weak,
not-yet-robust lead: Spearman rho=+0.095 (p=0.028) on a 5-day
look-back window, but rho drops to +0.057 (p=0.19) at 10 days and +0.004
(p=0.92) at 20 days, and the bucket pattern isn't a clean monotonic ramp
at any window. Not added as a filter; flagged as worth another look
(possible confound with the existing volume gate) but not pursued
further this session.

Full evidence trail above (params sweep, hold sweep, gap sweep, the
gap=3 mirage, and both the original and corrected full-101-ticker
validations) is what justifies treating this branch as a live-worthy
forward record despite the headline t-stat still sitting under this
project's usual t>2 bar in the no-overlay case (t=2.06) and only just
clearing it with the drawdown exit (t=2.40) -- see the "what made it
worth trading anyway" reasoning in `two_touch_low_live.py`'s docstring.

**Trade frequency: why ~51/yr, and an attempt to relax it (2026-09-22,
same day, later still).** User asked why the signal fires so rarely given
the pattern doesn't sound unique. Ran `diagnose()` over every (ticker,
day) in the full 101-ticker history (283,608 candidate days) to get the
exact funnel:

| stage | survives |
|---|---|
| is a fresh 5-day low at all | 20.8% |
| ...and birth-day volume > its own 10-day average | 68.8% |
| ...and prominence (prior day >=0.5% above the low) | 71.1% |
| ...and L+1 actually rebounds (pct1 > 0) | 53.5% |
| ...and rebound size is in the traded band | 49.7% |
| ...and rebound happens on *weak* volume | 46.0% |
| ...and retest lands back in [L, L+1] | 27.3% |
| ...and retest volume clears the floor | 58.6% |

No single gate is rare in isolation (several pass 50-70% of what reaches
them); seven roughly-independent gates compound multiplicatively to
~1-in-500 candidate days, which is ordinary arithmetic, not a bug. The
two tightest individual gates are the rebound's weak-volume requirement
(46.0%) and the retest landing back inside `[L, L+1]` (27.3%, the single
tightest in the chain).

Tested relaxing exactly those two gates on the 20-ticker seed=21 subset
(`two_touch_low_relaxed_variant.py`, `decide()` itself untouched -- a
separate experimental function): looser `rebound_volume_threshold`
(1.0->1.3->1.5x) and/or dropping the retest's upper bound at
`close[L+1]` entirely (only `close[L2] >= close[L]` still required).

| variant | n | excess | t |
|---|---|---|---|
| locked (baseline) | 106 | +0.07% | +0.24 |
| volume<1.3x, capped | 195 | -0.08% | -0.40 |
| volume<1.5x, capped | 225 | -0.20% | -1.03 |
| volume<1.0x, uncapped | 352 | -0.11% | -0.61 |
| volume<1.3x, uncapped | 655 | -0.14% | -0.95 |
| volume<1.5x, uncapped | 758 | -0.18% | -1.55 |

**Rejected.** Trade count rises up to 7x, but excess return degrades
monotonically with every relaxation, in every combination tried, no
exceptions. These two gates aren't incidental friction cutting down an
otherwise-good population -- they're doing real discriminating work;
loosening them mostly adds worse trades rather than revealing a larger
pool of good ones. Consistent enough (monotonic, no reversals) that a
full-101/fresh-cross-section run wasn't run to confirm further. If more
trade frequency is wanted, a larger ticker universe looks more promising
than loosening these two gates.

**Larger-universe attempt tested and rejected (2026-09-22, later same day).**
The relaxation section above ends "a larger ticker universe looks more
promising than loosening these two gates" -- that guidance turned out to
be wrong. Tested a candidate pool of 100 additional large-cap tickers
(S&P 100/500 names, multi-$B market cap, no penny stocks, sector-diverse,
disjoint from the live 101-ticker `LIVE_TICKERS` universe -- see
`strategies/five_day_bounce/experiments/two_touch_low_new100_check.py`),
using the exact same causal day-by-day harness and locked params/
retest_gap=2/DD_PCT=1% as the live script:

| | n | hit | excess | t |
|---|---|---|---|---|
| 5-day hold only | 470 | 60.0% | -0.16% | -0.88 |
| + market-drawdown exit | 470 | 54.3% | -0.08% | -0.47 |

Excess return flips negative -- not just weaker, the wrong sign -- on a
pool with more raw trade candidates than the live universe itself. This
is broad-based, not a few outliers dragging the mean: 52 of 97 tickers
with at least one trade have negative mean excess on their own, and the
median per-ticker excess across the pool is negative. **Rejected --
these 100 tickers were not added to the live run.** Same shape of finding
as [[reference-five-day-bounce-doc]]'s fresh-60-ticker flip: an edge that
looks solid on one ticker universe (full-101: t=+2.40; fresh-81 subset of
that same universe: t=+2.13) is not guaranteed to hold on a *different*
selection of large caps, even a large, sector-diverse, non-penny-stock
one. The original 101-ticker universe may simply not be a representative
sample of "large caps in general" for this signal. Trade frequency stays
at ~51/yr pooled across the validated 101; no lever tried so far (gate
relaxation, universe expansion) has raised it without destroying the edge.

**No-volume-gates + early-reversal-exit variant tested and rejected
(2026-09-22, later same day).** Another attempt at raising trade
frequency: drop the rebound_retest branch's two volume gates
(`rebound_volume_threshold` at L+1, `volume_threshold_buy` at L+2 --
Stage 1's own volume gate at L untouched, price band untouched) and add
a new exit -- bail out at L+3 if close[L+3] < close[L+2] (the buy day),
instead of waiting out the fixed 5-session hold
(`two_touch_low_novolume_earlyexit.py`). Four variants, isolated, on the
20-ticker seed=21 prototype subset:

| variant | n | hit | excess | t |
|---|---|---|---|---|
| locked (baseline) | 106 | 64.2% | +0.07% | +0.24 |
| + early-reversal exit only | 106 | 45.3% | -0.14% | -0.63 |
| no volume gates only | 390 | 59.0% | -0.15% | -0.93 |
| no volume gates + early exit | 401 | 36.9% | -0.29% | **-2.31** |

**Rejected.** Both changes hurt individually, and combined they're worse
than either alone -- not a case of two weak effects cancelling out or one
compensating for the other. The early-exit rule alone drops the hit rate
from 64% to 45% (it fires on 45 of 106 trades) for a t that goes from
+0.24 to -0.63; stacked on the volume-gate removal it fires on almost
half of all trades (194/401) and hit rate craters to 36.9%. Plausible
reading: the rebound_retest pattern's edge (to the modest extent it has
one) depends on the reversion taking more than one session to play out --
a same-day-plus-one down tick is ordinary noise for this setup, not
disconfirmation, so cutting there mostly removes trades before they get
the chance to work. Consistent enough at n=401 (t=-2.31, not a coin-flip
result) that a full-101/fresh-81 run wasn't run to confirm further, same
call as the earlier gate-relaxation rejection above.

**Back to basics: pure-price double-bottom, no volume at all, tested and
inconclusive-to-negative (2026-09-22, later same day).** Stripped every
volume knob out and rebuilt the original premise as literally as
possible: L1 (local low) -> H1 (a "smallish" local high after L1,
rise<=2%) -> L2 (another local low after H1) -> BUY on the first close
that breaks back above H1 ("confirmed by another high" -- the classic
double-bottom breakout, H1 as the neckline). Both extrema types are
backward-only rolling turning points (see local_low/local_high in
`two_touch_low_double_bottom.py`), same causality discipline as the rest
of this candidate, applied to highs for the first time. Fixed 5-day
hold, 10bps cost, no other filters. 20-ticker seed=21 prototype subset:

| | n | hit | net | excess | t |
|---|---|---|---|---|---|
| double-bottom, no volume | 620 | 51.1% | +0.12% | -0.09% | -0.66 |
| placebo (matched n, random entry) | 620 | -- | -- | -0.08% | -- |

**Causal lift over the placebo is -0.02%, i.e. nothing** -- these trades
don't behave differently from randomly timed entries with the same
trade count and hold. Trade frequency did rise a lot (620 trades / 20
tickers / ~11yr ~= 2.8/ticker/yr, several times the locked rule's rate),
but there's no edge left to spread across the extra trades. Year-by-year
excess flips sign with no visible trend (+0.99, -1.07, -0.24, -1.22,
+0.52, -1.00, +1.20, +0.07, -1.11, +0.60, -1.85, +0.38) -- looks like
noise, not a weakening-but-real effect.

**Open design question, not yet tested:** this version doesn't require
L2 to land near L1's level -- "another local low" was left unconstrained
per the literal request, so the pattern currently accepts any
low-high-low sequence, not specifically a *retest of the same level*
(which is what "two-touch" originally meant back in the tech_levels
world). Whether constraining L2 to within some tolerance of L1 changes
this result is untested -- flagged for next session rather than guessed
at, since it changes what the pattern claims to be measuring.

**Fixed-offset same-level retest, no volume, tested and rejected
(2026-09-22, later same day).** Third no-volume rebuild, tighter spec:
L (a "distinguishable" low -- close[L] at least `fall_pct`=3% below the
highest close in a 10-day lookback, replacing the old spec's token
0.5%-vs-prior-bar prominence check) -> L+1 (unconstrained, "some
wiggle") -> L+2 (the real second touch: close[L+2] in
[close[L], close[L]*1.01] -- close to L's level but not below it) -> buy
at L+3's close (one session of execution lag past the signal). Hold
swept over {2,3,4,5} sessions, 20-ticker seed=21 prototype
(`two_touch_low_distinguishable_retest.py`):

| hold | n | hit | excess | t |
|---|---|---|---|---|
| 2 | 1199 | 52.0% | -0.14% | -2.30 |
| 3 | 1154 | 53.4% | -0.17% | -2.23 |
| 4 | 1106 | 54.8% | -0.11% | -1.23 |
| 5 | 1081 | 55.5% | -0.09% | -0.87 |

**Rejected -- excess is negative at every hold tested**, significantly so
at hold=2/3 (t=-2.30, -2.23; n>1100 each, not a small-sample fluke). Even
the least-bad case (hold=5) only ties a matched random-entry placebo
(placebo excess -0.15% vs strategy -0.09%, lift +0.06% -- both numbers
negative, the "lift" is one negative being slightly less negative than
another). No hold length turns this into a real edge.

**Reading, after three no-volume rebuilds now (this one, the free
double-bottom above, and the earlier no-volume+early-exit combo):** every
version that drops the L+1/L+2 volume gates comes back flat-to-negative,
while the original volume-gated rule holds up at t=+2.40 (full-101) and
+2.13 (fresh-81). That's a consistent enough pattern across three
independently-shaped attempts to treat "the volume gates are load-bearing,
not incidental" as the working conclusion for this candidate, not an
open question -- matches the funnel breakdown's own earlier finding that
the two volume gates were the two tightest filters in the chain (46.0%
and 58.6% survival) and gate-relaxation already failed once before this.
If more trade frequency is still wanted, changing the *shape* of the
price pattern without volume has now been tried three ways and hasn't
worked; a volume-based change (a different threshold, a different
volume definition) looks like the more promising remaining direction,
not another volume-free variant.

**Same-level retest, confirmed above L throughout -- better, still not
an edge (2026-09-22, later same day).** Follow-up to the rejected
fixed-offset retest above: added two more price-only checks -- L+1 must
close above L, and L+3 (the buy day) must also close above L, on top of
the existing distinguishable-low and same-level-retest conditions
(`two_touch_low_retest_confirmed.py`). Same 20-ticker prototype, hold
swept over {2,3,4,5}:

| hold | n | hit | excess | t |
|---|---|---|---|---|
| 2 | 546 | 52.0% | -0.12% | -1.50 |
| 3 | 544 | 52.6% | -0.13% | -1.32 |
| 4 | 536 | 55.6% | -0.07% | -0.57 |
| 5 | 527 | 54.8% | **+0.01%** | **+0.10** |

Trade count roughly halved from the unconfirmed version (the two new
checks are real filters), and hold=5's excess moved from clearly
negative to statistically indistinguishable from zero (t=+0.10) --
better than every prior no-volume attempt today, but **t=0.10 is noise,
not a result.** Causal lift over a matched placebo is +0.14% (placebo
-0.12% vs strategy +0.01%), which sounds like an improvement but is one
near-zero number minus a negative one -- fragile, not a discovery.
Shorter holds (2, 3) are still significantly negative. Year-by-year is
noisy with small per-year samples (11-64) and no visible trend.

**Where this leaves things:** requiring the whole pattern to stay above
L (not just the L+2 retest) is the first no-volume variant today that
isn't outright negative, which is progress, but "flat" still falls well
short of this candidate's own t>2 bar (the volume-gated original: +2.40
full-101, +2.13 fresh-81). Worth a full-101/fresh-81 run before
concluding further, given the small-subset t is close to zero either
way and a 20-ticker read has limited power to distinguish "no effect"
from "small positive effect" -- not run yet, flagged for next step.

**Published doc (2026-09-22):** "The Two-Touch Low" --
https://claude.ai/artifact/KMbHd4PkcEqzv5knd1iX7V -- premise, the rule,
the full results trail (baseline, param sweep, hold sweep, the
retest_gap=3 mirage, full-101 validation), live status, code chain, and
open items. Private (owner-only); share via the page's share menu. This
file and `strategies/two_touch_low/NOTES.md` stay the source of truth if
the link ever breaks.

**Open items to pick up next session:**
- **Scheduling not decided.** `run_live_log_near_close.sh` /
  `run_live_log.sh` exist and both run cleanly by hand (tested
  2026-09-22), but no launchd plist has been created or loaded -- nothing
  runs automatically yet. Decide whether to wire up the same
  `~/Library/LaunchAgents` pattern the old strategy used
  ([[project-continuation-execution-timing]]) at 21:40 local, and whether
  to also schedule the 08:00 settled-price companion.
- **Near-close volume caveat unresolved.** The buy gate depends on
  *today's* volume ratio (the retest day), which near-close is a partial,
  still-forming figure that understates the settled value -- biases
  toward missed signals, not false ones (see
  `two_touch_low_live.py`'s module docstring). Open question: accept this
  as-is (same spirit as the old system's near-close price approximation),
  or have entries for this signal specifically wait for the settled
  08:00 run instead of firing near-close.
- **Forward record is empty so far** (`strategies/two_touch_low/data/trades.csv`
  has no closed trades yet, run started 2026-09-22) -- per
  [[validation-first-quant-work]], the forward record is the only true
  out-of-sample evidence and needs time to accumulate before it says
  anything.

**Causality check plan:** every forward reference here is bounded to
L+1 or L+2, and the buy is dated at the completing bar, not backdated --
structurally different from the old `find_peaks`-based bug, which had an
effectively unbounded confirmation window. Still worth running through
the day-by-day truncate-and-rebuild harness
(`strategies/five_day_bounce/experiments/original_rule_daybyday.py`
pattern) as a cheap confirmation rather than trusting the reasoning alone.

**Close-location-value (clv) on the low day, tested and rejected
(2026-09-23).** User hypothesis: a qualifying low at day L is more
likely to bounce if L itself closed closer to its own high than its own
low (hammer/pin-bar absorption -- buyers took the day back before the
close). Quantified as `clv[L] = (close[L]-low[L])/(high[L]-low[L])` in
[0,1], a different axis from the existing `volume_ratio[L]` gate (side
of the day that won, not how much activity happened).
Correlation-first test, not a new gate: took the existing Stage 1
qualifying-low population unchanged (backward rolling min, prominence,
volume gate -- `two_touch_low_daybyday.decide()`'s first three checks),
computed `clv[L]` for each, and correlated (Spearman) against two
outcomes measured directly from L, bypassing the Stage 2 buy-trigger
machinery entirely: `pct1` (next-day move) and `fwd5` (5-day forward
return, the strategy's own hold length). 20-ticker seed=21 prototype
subset, 2015-2026 (`two_touch_low_clv_correlation.py`):

| | rho | p | n |
|---|---|---|---|
| clv vs pct1 | -0.022 | 0.092 | 5755 |
| clv vs fwd5 | **-0.038** | **0.004** | 5755 |
| clv vs volume_ratio[L] (confound check) | +0.046 | 0.001 | 5755 |

**Rejected -- and the significant one runs opposite to the hypothesis.**
Quantile-bucketed fwd5 falls roughly monotonically from Q1 (close near
the day's *low*, clv small) at +0.71% to Q5 (close near the day's
*high*, clv large) at +0.17% -- weak-closing low days bounced *more*,
not less, over the next 5 sessions. Year-by-year: no single year clears
p<0.05 in the hypothesized (positive) direction; the two years that do
clear p<0.05 (2018 rho=-0.087, 2019 rho=-0.136) are both negative, the
rest flip sign near zero with no visible trend. Confound check rules out
"this is volume in disguise" -- clv and volume_ratio[L] are barely
correlated (rho=+0.046) -- so whatever the (tiny) effect is, it's
independent of the existing volume gate, just not usefully predictive
either way. Effect sizes throughout are small (|rho|<0.04); this reads
as "no signal here," same shape of outcome as the earlier "bigger fall +
high volume" side test above, not as a confirmed reversed-sign effect to
chase further.

**Two-touch-low v2: from-scratch redesign, fully tested and rejected
(2026-09-23).** A complete rewrite of the two-touch-low premise, kept as
an independent candidate rather than a variant of the live
`rebound_retest` rule (`two_touch_low_daybyday.py`) -- nothing here
touches or risks that rule's own forward record.

**Spec.** Stage 1 -- first trough at day L: `close[L] < close[L-n_back..L-1]`
and `close[L] < close[L+1..L+n_fwd]` (a two-sided local min, backward
`n_back` in [5,10] and forward `n_fwd` in [1,5], both swept), confirmed
only as of the close of L+n_fwd. Gate: `buyers_took_over[L] OR
volume_ratio[L] > volume_threshold_L`, where
`buyers_took_over[j] = close[j]-low[j] > high[j]-close[j]` (close above
the day's own high-low midpoint) and `volume_ratio` is the existing
trailing-10-day definition. Stage 2 -- second trough at the *fixed*
offset `L2 = L + n_fwd + 1` (not a scanned window -- this is what keeps
the forward-looking Stage 1 check causal: by L2's close, days
L+1..L+n_fwd have already happened), requiring
`|close[L2]-close[L]|/close[L] <= close_tolerance` (swept up to 1%) and
the same buyers_took_over-or-volume gate. Supersession: a newer confirmed
trough strictly between L and L2 discards L outright (it gets its own
turn when the day-by-day loop reaches it) -- falls naturally out of the
loop structure, since only L+1 is ever confirmable in time to matter for
a given L. Buy fires at close[L2]; exit is a 5-day hold or SPY
market-drawdown (dd_pct=1%), same convention as the live v1 rule.
Implementation: `two_touch_low_v2_daybyday.py`, causality verified by the
same truncate-and-recompute assertion as the rest of this family -- no
leakage found at any sample size tried. `pull_all()`
(`tech_level_live.py`) was extended with an `include_hl=False` flag to
carry high/low alongside close/volume -- off by default, so none of the
~30 existing callers changed behavior.

**Sweep (`two_touch_low_v2_sweep.py`), 20-ticker seed=21 prototype,
one-at-a-time.** Loose starting point (n_back=7, n_fwd=3,
volume_threshold_L=volume_threshold_L2=1.0, close_tolerance=0.5%): n=107,
t=+0.85. `n_back`'s apparent optimum first looked like it sat at the edge
of the tested grid (best at n_back=5 of [5..10]) -- widened to include
3-4 and it resolved into a real interior peak (n_back=5 alone: n=124,
t=+1.50, flanked by lower t at 3/4/6), not a grid-edge artifact. Combined
one-at-a-time winners across all five variables (n_back=5, n_fwd=3,
volume_threshold_L=1.0, volume_threshold_L2=0.8, close_tolerance=0.5%):
n=182, t=+1.50 (same t as the n_back=5 row by coincidence -- different
population) -- this became "combined-best" for everything that follows,
with the same caveat `two_touch_low_sweep.py` already carries for the v1
rule: stacking one-at-a-time winners isn't a validated joint result.

**Chart review (`two_touch_low_v2_charts.py`) surfaced a hypothesis, and
checking it directly showed the opposite.** A visual sample of 10
combined-best signals suggested losers clustered where L2's close sat
near its own day's low rather than its high -- the reverse of what
`buyers_took_over` selects for. Breaking the 182 combined-best trades
down by which OR-branch actually fired at L2
(`two_touch_low_v2_bto_check.py`): `volume_only` (n=126, the dominant
bucket) was the *best*-performing (t=+1.68); `bto_only` (n=14) was the
single *worst* bucket in the whole combo (t=-2.06). A clean BTO-only
variant (both volume thresholds set to +inf, so only the candle
condition gates entry) collapsed to n=17, t=-0.83 -- worse than the full
combo, not better.

**Refined into position terciles -- reversed-sign result, independently
at both L and L2.** Replaced the binary `buyers_took_over` test with a
continuous `position = (close-low)/(high-low)` and bucketed into thirds
on the *ungated* population (Stage 1 + Stage 2 only, both volume
thresholds at -inf so the OR gate always passes regardless of candle
shape -- the clean population to slice by position after the fact, n=397;
`two_touch_low_v2_position_terciles.py`). At both L and L2, independently:
bottom_third (close near the day's *low*) was the best bucket (L: n=256,
t=+1.92; L2: n=249, t=+0.90); top_third (close near the day's *high* --
the original `buyers_took_over` condition) was the worst (L: t=-1.14;
L2: t=-0.50). A strict top-third-at-both gate produced n=4, all losers.
This matches the sign of the close-location-value finding on the *v1*
rule's L day earlier in this same file (2026-09-23, same day) -- weak-
into-the-close troughs outperforming strong-into-the-close ones is now a
pattern seen independently on two differently-constructed candidates, not
an artifact of this spec's particular construction. Chart spot-check
(`two_touch_low_v2_bottom_third_charts.py`) confirmed the buckets look
structurally ordinary, not mislabeled -- bottom-third entries just have a
weak-into-the-close candle on the relevant day(s), win more often than
not, but far from a sure thing.

**Volume alone, at L2, beat every other cut tried in-sample.**
`volume_ratio[L2] > 1.0` alone (`two_touch_low_v2_l2_volume_cross.py`):
n=153, t=+2.09 -- the best single number of the whole exploration,
beating the position tercile, the original BTO gate, and their
combination (`bottom_third & vol_L2>1.0`: n=90, t=+1.74).

**None of it replicated on the 81 LIVE_TICKERS never touched during
tuning** (same fresh-81 discipline as the v1 rule's validation --
`two_touch_low_v2_l2_volume_freshcheck.py`,
`two_touch_low_v2_freshcheck_combined.py`). Every cut got worse, several
reversed sign entirely:

| cut | in-sample (20 tickers) | fresh-81 |
|---|---|---|
| vol_L2>1.0 alone | n=153, t=+2.09 | n=677, t=-0.72 |
| vol_L2<=1.0 alone | n=244, t=-0.54 | n=934, **t=-3.14** |
| bottom_third (L2) alone | n=249, t=+0.90 | n=963, t=-0.72 |
| bottom_third & vol_L2>1.0 | n=90, t=+1.74 | n=391, t=+0.44 |
| **combined-best combo (the real OR-gated rule)** | n=182, t=+1.50 | **n=768, t=-3.09** |

The combined-best combo -- the actual rule, not a diagnostic cut -- does
not just fail to replicate, it comes back a statistically significant
*loser* on the larger sample (t=-3.09, n=768, hit=48.8%, excess=-0.301%),
negative in 6 of the last 7 years (2020-2026, only 2019 clearly
positive). The fresh-81 `vol_L2<=1.0` "control" bucket also came back
significantly negative (t=-3.14, n=934) rather than neutral, suggesting
the bare Stage 1 + Stage 2 pattern may carry a real negative tilt on the
broader universe that no gate tried today fixed.

**Rejected.** Everything explored today -- the OR gate, the BTO-only
variant, the position terciles, the volume>1.0 gate, and their
combinations -- was fit on the single 20-ticker seed=21 draw and none of
it held up on the 81 tickers withheld from tuning. Per
[[validation-first-quant-work]]: reported as a clean negative, not
soft-pedaled as "needs more tuning" -- patching individual gates on a
result this far underwater on the larger sample would likely just refit
the same noise differently. The v1 `rebound_retest` rule is unaffected;
this was an independent redesign attempt, not a modification of it.

Code (all in `strategies/five_day_bounce/experiments/`):
`two_touch_low_v2_daybyday.py`, `two_touch_low_v2_sweep.py`,
`two_touch_low_v2_charts.py`, `two_touch_low_v2_bto_check.py`,
`two_touch_low_v2_position_terciles.py`,
`two_touch_low_v2_bottom_third_charts.py`,
`two_touch_low_v2_l2_volume_cross.py`,
`two_touch_low_v2_l2_volume_freshcheck.py`,
`two_touch_low_v2_freshcheck_combined.py`.

## Mean reversion after a big selloff -- tested and rejected (2026-09-23)

New candidate direction, independent of the abandoned two-touch-low
family: instead of "a fresh local low, however marginal," require actual
*magnitude* -- a causal drawdown-depth z-score (candidate #2 above),
generalized so "deep dip" means the same thing across tickers/regimes:

```
roll_max[i] = max(close[i-N:i])   (trailing N days, excludes day i)
roll_std[i] = std(close[i-N:i])   (trailing N days, excludes day i)
z[i] = (close[i] - roll_max[i]) / roll_std[i]
```

Fires when `z[i] <= -z_thresh`. Built in from the start (not bolted on
after a baseline, per this session's explicit plan): a market-wide vs.
idiosyncratic split using the identical z-score on SPY over the same
window -- `market_wide` when SPY itself is deep in its own drawdown
(hypothesis: forced-selling/liquidity, plausibly reverts), `idiosyncratic`
when SPY is roughly flat (hypothesis: name-specific bad news, plausibly
doesn't) -- a middle "mixed" zone excluded from both. Buy at close[i]
(the day the trigger completes), fixed 5-day hold, 10bps cost, one
position per ticker. Causality verified by the same truncate-and-rebuild
assertion as the rest of this file -- no leakage found.

**Price-only grid, 20-ticker seed=21 prototype
(`selloff_bounce_sweep.py`):**

| n_window | z_thresh | n | excess | t | market_wide t | idiosyncratic t |
|---|---|---|---|---|---|---|
| 10 | 2.0 | 4630 | -0.062% | -1.19 | -0.93 | -0.77 |
| 10 | 2.5 | 3800 | -0.068% | -1.19 | -1.17 | -0.36 |
| 10 | 3.0 | 2989 | -0.016% | -0.24 | -0.52 | +0.40 |
| 20 | 2.0 | 4341 | -0.085% | -1.61 | -1.43 | -0.77 |
| 20 | 2.5 | 3514 | -0.078% | -1.29 | -0.70 | -1.34 |
| 20 | 3.0 | 2584 | -0.109% | -1.54 | -1.19 | -1.00 |
| 30 | 2.5 | 3401 | -0.135% | -2.28 | -1.60 | -1.76 |
| 30 | 3.0 | 2483 | -0.099% | -1.37 | -0.89 | -1.21 |

**Every combination has negative raw excess return**, tightening the
threshold does not trend toward positive (worst t is the *tightest*
setting tested, 30-day/2.5sigma), and the market-wide/idiosyncratic split
does not separate as hypothesized -- market_wide is consistently negative
right alongside idiosyncratic, sometimes worse. Matches this file's
established pattern for the two-touch-low family: bare price patterns
(double-bottom, fixed-offset retest, retest-confirmed) all came back
flat-to-negative while the volume-gated version held up.

**Capitulation-volume gate, quantile-bucketed (not threshold-swept, per
this file's own "Reminder for later sweeps" warning against nested-subset
artifacts) on the fixed n_window=10/z_thresh=2.5 trigger population
(`selloff_bounce_volume_quantile.py`):**

| bucket (volume_ratio[trigger day]) | n | excess | t |
|---|---|---|---|
| Q1 (lowest) | 760 | +0.090% | +0.79 |
| Q2 | 760 | -0.120% | -0.97 |
| Q3 | 760 | -0.090% | -0.72 |
| Q4 | 760 | -0.117% | -0.88 |
| Q5 (highest) | 760 | -0.105% | -0.73 |

Spearman(vr, excess_ret) rho=-0.015, p=0.358, n=3800 -- no dose-response
at all, and no monotonic gradient across buckets in either direction.
Splitting further by branch surfaces one cell worth flagging but not
chasing: idiosyncratic + lowest-volume quintile, n=214, hit=68.7%,
t=+1.93 -- runs *opposite* the capitulation-volume hypothesis (low
volume, not high) and is 1 of 10 cells sliced here on top of the 8 grid
points already tried above, so per
[[validation-first-quant-work]] this reads as a multiple-comparisons
artifact, not a discovery -- not investigated further this session.

**Rejected.** Two independent formulations of the same premise (bare
price magnitude; magnitude + capitulation volume) both came back
flat-to-negative on the 20-ticker prototype, with no fresh-81 validation
needed since neither cleared the prototype-stage bar to begin with. The
mechanism hypothesis behind the SPY split (market-wide selloffs
mean-revert, idiosyncratic ones don't) also did not show up in either
formulation. Reported as a clean negative rather than soft-pedaled as
"needs more tuning," per [[validation-first-quant-work]].

Code (all in `strategies/five_day_bounce/experiments/`):
`selloff_bounce_daybyday.py`, `selloff_bounce_sweep.py`,
`selloff_bounce_volume_quantile.py`.

**Streak-based magnitude (candidate #3), tested and rejected, same
session.** Alternative way of sizing "big" -- instead of a drawdown-depth
z-score vs. the recent high, size the selloff by the current
consecutive-down-day streak (length or cumulative decline), z-scored
against that ticker's own trailing streak-severity distribution
(`shift(1).rolling(dist_window)` baseline, same causal construction as
Phase 0/1's rolling_z). Both variants the notes explicitly named --
`mode="length"` and `mode="decline"` -- tested side by side; SPY
market-wide/idiosyncratic split, buy timing, exit, and cost held
identical to Phase 0/1 for a clean comparison
(`selloff_bounce_streak_daybyday.py`, `selloff_bounce_streak_sweep.py`).

| dist_window | z_thresh | mode | n | excess | t | market_wide t | idiosyncratic t |
|---|---|---|---|---|---|---|---|
| 60 | 1.5 | length | 2610 | -0.129% | -1.89 | -0.98 | -1.87 |
| 60 | 2.0 | length | 1833 | -0.133% | -1.60 | -1.15 | -1.13 |
| 120 | 1.5 | length | 2346 | -0.190% | -2.55 | -1.16 | -2.89 |
| 120 | 2.0 | length | 1616 | -0.064% | -0.71 | -0.13 | -1.13 |
| 60 | 1.5 | decline | 2396 | -0.076% | -1.03 | -0.02 | -2.00 |
| 60 | 2.0 | decline | 1790 | -0.108% | -1.29 | -1.01 | -0.87 |
| 60 | 2.5 | decline | 1309 | -0.156% | -1.62 | -1.43 | -0.79 |
| 120 | 1.5 | decline | 2150 | -0.139% | -1.78 | -1.12 | -1.77 |
| 120 | 2.0 | decline | 1546 | -0.102% | -1.07 | -1.02 | -0.35 |
| 120 | 2.5 | decline | 1141 | -0.143% | -1.31 | -1.51 | +0.22 |

**Every one of 10 combinations across both modes has negative raw
excess**, no trend toward positive with a tighter threshold, and the
market-wide/idiosyncratic split again fails to separate consistently in
either direction (idiosyncratic sometimes worse than market_wide, e.g.
120/1.5/length: -2.89 vs -1.16; sometimes the reverse, e.g. 60/1.5/decline:
-2.00 vs -0.02 -- no stable pattern across the grid). **Rejected**, third
independent formulation of the same "big selloff" premise (bare drawdown
z-score; + capitulation volume slice; streak length/decline z-score) to
come back flat-to-negative on the 20-ticker prototype. Causality verified
by the same truncation assertion as the rest of this file -- no leakage
found, this is a real result on the specified population, not a bug.

**Reading, three formulations in:** the market-wide-selloff-mean-reverts
mechanism hypothesis behind the SPY split has not shown up in any
formulation tried so far, and "how big" (z-score vs. recent high, streak
length, streak decline) doesn't seem to be the missing ingredient either
-- all three read as noise around a small negative, not variations
converging toward a real edge from different angles. Per
[[validation-first-quant-work]], this is reported as the working
conclusion for the session, not soft-pedaled as "needs more tuning."

**Selloff + sideways consolidation before buying, tested and rejected,
same session.** Fourth formulation: instead of buying immediately at the
selloff day (Phase 0/1/3 all did), require the price to go sideways
first -- add a Stage 2 requiring close[L+1..L+consol_days] to ALL sit
within a +/-band_pct band around close[L] (no breakout either direction),
buy at close[L+consol_days] only if every one of those days held inside
the band. Tested against BOTH selloff definitions (drawdown-depth
z-score and streak z-score) unchanged from above, band_pct fixed at 2%
(middle of the 1-3% range considered), consol_days swept {1,2,3,5}, hold
swept {3,5,10} -- 24 combinations total
(`selloff_bounce_consol_daybyday.py`, `selloff_bounce_consol_sweep.py`):

| type | consol | hold | n | excess | t |
|---|---|---|---|---|---|
| drawdown | 1 | 3 | 4177 | -0.119% | -3.13 |
| drawdown | 1 | 5 | 3394 | -0.102% | -1.77 |
| drawdown | 1 | 10 | 2471 | -0.068% | -0.72 |
| drawdown | 2 | 3 | 3351 | -0.125% | -3.05 |
| drawdown | 2 | 5 | 2788 | -0.118% | -1.92 |
| drawdown | 2 | 10 | 2086 | -0.133% | -1.32 |
| drawdown | 3 | 3 | 2640 | -0.114% | -2.51 |
| drawdown | 3 | 5 | 2264 | -0.124% | -1.91 |
| drawdown | 3 | 10 | 1750 | -0.160% | -1.48 |
| drawdown | 5 | 3 | 1667 | -0.183% | **-3.42** |
| drawdown | 5 | 5 | 1474 | -0.086% | -1.16 |
| drawdown | 5 | 10 | 1197 | -0.128% | -1.07 |
| streak | 1 | 3 | 1358 | -0.168% | -2.39 |
| streak | 1 | 5 | 1263 | -0.251% | -2.77 |
| streak | 1 | 10 | 1094 | -0.129% | -0.93 |
| streak | 2 | 3 | 1039 | -0.190% | -2.82 |
| streak | 2 | 5 | 970 | -0.167% | -1.78 |
| streak | 2 | 10 | 871 | -0.120% | -0.87 |
| streak | 3 | 3 | 781 | -0.176% | -2.43 |
| streak | 3 | 5 | 749 | -0.157% | -1.54 |
| streak | 3 | 10 | 687 | -0.085% | -0.53 |
| streak | 5 | 3 | 450 | -0.130% | -1.35 |
| streak | 5 | 5 | 447 | -0.061% | -0.50 |
| streak | 5 | 10 | 426 | +0.125% | +0.66 |

**23 of 24 combinations negative.** The one positive cell (streak,
consol=5, hold=10: n=426, t=+0.66) is weak, not significant on its own,
and its neighbors (same consol_days=5, shorter holds) are still
negative -- reads as drift toward zero as hold lengthens (diluting any
timing-specific effect toward plain market beta) rather than a real
effect, especially set against 23 other negative cells in the same grid
(multiple-comparisons exposure). Adding the consolidation stage did not
rescue either selloff definition -- several cells got *more* negative
than the bare trigger (drawdown/consol=5/hold=3: t=-3.42, the single
worst result of this entire session). **Rejected.**

**Reading, four formulations in:** bare magnitude (drawdown z-score,
streak z-score), a capitulation-volume slice, and now a sideways-
consolidation entry filter have all been tried on top of the same "big
selloff" premise, independently, and all come back flat-to-negative.
Nothing tested so far turns this into a real edge; per
[[validation-first-quant-work]] this is reported as the working
conclusion for the premise as currently framed, not as "needs more
tuning."

**CLV direction on the trigger day, tested and rejected (same session,
later).** Correlation-first, same methodology as the earlier two-touch-low
CLV test: took the Stage 1 trigger population unchanged (both selloff
definitions), computed `clv[L] = (close[L]-low[L])/(high[L]-low[L])`, and
correlated against raw pct1/fwd5 (`selloff_bounce_clv_correlation.py`).
Raw correlations looked promising and directionally consistent with the
two-touch-low finding (weak-into-the-close outperforms): drawdown
rho=-0.027 (p=0.003, n=11784), streak rho=-0.088 (p<0.001, n=2986), with
a near-monotonic bucket spread on the streak population (raw fwd5: Q1
+1.02% -> Q5 -0.17%).

Per [[validation-first-quant-work]], that raw correlation was broken
before being reported: re-run through the actual backtest machinery
(busy-ticker-deduped trades, cost-net excess over the equal-weight
benchmark, placebo, quantile-bucketed t-stats --
`selloff_bounce_clv_gate.py`) instead of raw fwd5 on an unconstrained
population. Most of it evaporated:

| | Spearman rho | p | n |
|---|---|---|---|
| drawdown | -0.012 | 0.445 | 3800 |
| streak | -0.042 | 0.099 | 1546 |

Drawdown: not significant at all, bucket pattern not monotonic (Q1
-0.038%, Q2 +0.035%, Q3 -0.140%, Q4 +0.054%, Q5 -0.254%). Streak: what
survives is asymmetric -- Q4/Q5 (strong close, near the day's high) are
significantly bad (t=-2.14, t=-1.75), but Q1 (weak close) is not
significantly good (t=+0.41) -- an avoid-signal on strong closes, not a
buy-signal on weak ones. A concrete low-clv gate (bottom half of the
population) is not significantly positive either way: drawdown t=-0.56
(still negative), streak t=+0.36 (n=773, far from significant).
**Rejected** -- the raw-fwd5 correlation that looked like a real echo of
the two-touch-low finding did not survive being measured the same way
everything else in this file is measured.

**Cross-sectional rank reversal (candidate #8), tested and rejected,
same session.** A genuinely different mechanism from everything above --
rank each ticker's trailing n_return-day return against the *same
20-ticker peer universe* on that date (causal percentile rank,
`ret.rank(axis=1, pct=True)`, no reference to the ticker's own history at
all), buy the worst-ranked names, no SPY split (the premise is already
relative-to-peers by construction)
(`selloff_bounce_crosssectional_daybyday.py`,
`selloff_bounce_crosssectional_sweep.py`). Grid over n_return in
{5,10,20}, bottom_pct in {5%,10%,20%}, hold in {5,10} -- 18 combinations:

| n_return | bottom_pct | hold | n | excess | t |
|---|---|---|---|---|---|
| 5 | 5% | 5 | 1204 | -0.041% | -0.34 |
| 5 | 5% | 10 | 1049 | -0.144% | -0.82 |
| 5 | 10% | 5 | 2113 | -0.048% | -0.55 |
| 5 | 10% | 10 | 1740 | -0.123% | -0.95 |
| 5 | 20% | 5 | 3617 | -0.020% | -0.33 |
| 5 | 20% | 10 | 2719 | -0.032% | -0.33 |
| 10 | 5% | 5 | 976 | +0.103% | +0.82 |
| 10 | 5% | 10 | 750 | +0.064% | +0.34 |
| 10 | 10% | 5 | 1741 | -0.040% | -0.43 |
| 10 | 10% | 10 | 1269 | +0.038% | +0.27 |
| 10 | 20% | 5 | 3102 | -0.046% | -0.70 |
| 10 | 20% | 10 | 2127 | +0.077% | +0.71 |
| 20 | 5% | 5 | 809 | +0.063% | +0.44 |
| 20 | 5% | 10 | 595 | -0.002% | -0.01 |
| 20 | 10% | 5 | 1505 | +0.067% | +0.67 |
| 20 | 10% | 10 | 1037 | +0.061% | +0.37 |
| 20 | 20% | 5 | 2722 | -0.057% | -0.81 |
| 20 | 20% | 10 | 1775 | -0.015% | -0.13 |

**Rejected, but a different shape of negative than everything else in
this section** -- unlike the own-history formulations (significantly
negative in most cells), the cross-sectional version never comes back
significantly negative anywhere, but it also never clears a meaningful
positive bar: best cell is t=+0.82 (n_return=10, bottom_pct=5%, hold=5,
n=976), well within noise range given 18 cells tested. Reads as flat/
no-signal rather than a real edge in either direction; no fresh-81 run
needed since nothing cleared even a loose in-sample bar.

**Session-closing reading, six formulations in:** bare magnitude (two
definitions), + capitulation volume, + sideways consolidation, + CLV
direction, and a cross-sectional reframing have all now been tried
against "mean reversion after a big selloff," independently, covering
own-history magnitude, peer-relative magnitude, volume, candle shape, and
entry timing. Four came back significantly negative, one broke on
closer scrutiny, one came back flat. None produced a usable positive
edge. Per [[validation-first-quant-work]], this is the closing verdict
for the premise as explored this session, reported plainly rather than
left open as "needs more tuning."

Code (all in `strategies/five_day_bounce/experiments/`):
`selloff_bounce_streak_daybyday.py`, `selloff_bounce_streak_sweep.py`,
`selloff_bounce_consol_daybyday.py`, `selloff_bounce_consol_sweep.py`,
`selloff_bounce_clv_correlation.py`, `selloff_bounce_clv_gate.py`,
`selloff_bounce_crosssectional_daybyday.py`,
`selloff_bounce_crosssectional_sweep.py`.

## Suggested starting point

Run #1 (rolling N-day low) through the day-by-day harness at a couple of N
values, alongside #5 or #6 as an independent-mechanism placebo from the
start. If #1 doesn't beat that placebo, that's the "generic reversal, not
lows specifically" finding again — caught before anything is built on it,
not after.
