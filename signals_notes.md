# Sparse signals — session notes

**Status as of 2026-07-21.** Harness built (`events.py`) and working. First
signal tested (`signals.idio_shock`) and **refuted — both arms**. The up arm
looked like the strongest result in the project until it was run on 60 tickers
outside the original 40, where it flipped sign. Read the OOS section before
trusting anything in the in-sample section.

---

## Why this design

The two-stage forecast/portfolio design assumed edge is spread thinly across all
days. An IC of 0.04 measured over every day is equally consistent with a real
edge on 2% of days and nothing on the other 98% — in which case averaging over
all days is the wrong estimator and conditioning on "the signal fired" is the
right one.

The cost is sample size. Distinguishing a 1.5% mean excess from zero when
per-trade noise is ~6% needs ~64 trades for t = 2; a signal firing 20×/year takes
three years to reach a t-stat that means nothing after selection. Two mitigations,
and every signal here should use both: **pool across the 40 names** so events
accumulate 40× faster, and **take hypotheses with priors from outside this data**
so we estimate a magnitude on a known sign rather than discovering an effect.

### The arithmetic that constrains the design

Tilting one name by Δw for 10 days from an equal-weight base:

| tilt | excess EV/trade | contribution/trade | at 20 fires/yr | at 60 fires/yr |
|---|---|---|---|---|
| 5pp | 1.5% | 7.5bp | +1.5%/yr | +4.5%/yr |
| 15pp | 1.5% | 22bp | +4.5%/yr | +13%/yr |

**Costs are not the binding constraint** — a 5pp round trip at 10bps is 0.5bp
against 7.5bp of EV, 15:1. The binding constraints are EV *estimation error* and,
for a long-only book, concentration.

---

## The harness (`events.py`)

Contract: a signal returns `fires[stock, fire_date, side, strength]`; the harness
does everything else, identically for every signal, so none gets a bespoke metric
that flatters it.

- **`entry_lag=1`.** Entry is the close of the day *after* the fire. The signal
  comes off a close, the alert goes out after the close, the trade happens next
  session. Entering at the fire-day close is free lookahead.
- **`excess_ret` over equal weight** is the column to judge, never `ret`. A signal
  that fires in bull markets looks wonderful on raw return and adds nothing.
- **`block_bootstrap`** resamples calendar quarters, not individual trades.
  Holding windows overlap and a shock day fires several names at once, so the
  i.i.d. standard error is wrong. Report `t_block`, not `t_naive`.
- **`placebo`** draws random entry dates *on the same stocks*, matched on trade
  count and exit rule. This is the test that killed the technical-level overlay,
  which was mostly one name's drift.
- **`take_profit` / `stop_loss`** are checked on closes only. When both trigger on
  the same close the stop is assumed — the conservative read.
- Trades whose full window would run past the end of the series are **dropped, not
  truncated**, so the sample isn't biased toward whatever the last days did.

---

## Signal 1 — idiosyncratic shock (`signals.idio_shock`)

Residual return `r_i − β_i·r_m` (β on 252d), scaled by its own 66d trailing
volatility with `.shift(1)` so the shock day is outside its own denominator.
Vetoed when the market itself moved > 1σ. Repeat fires within 10 trading days on
the same name are deduped, or a multi-day selloff inflates `n` while adding no
information.

### The stated hypothesis is dead

Short-horizon reversal after a large idiosyncratic **drop**: flat everywhere.

| z ≤ | h=5 | h=10 | h=20 | h=40 |
|---|---|---|---|---|
| −2.0 | −0.14% | −0.05% | −0.10% | −0.14% |
| −3.0 | −0.07% | −0.09% | +0.01% | −0.39% |
| −4.0 | −0.23% | −0.23% | −0.51% | −1.10% |

Not one cell is meaningfully positive, and the trend runs the wrong way — bigger
drops are followed by *more* underperformance, not reversal.

### The mirror arm carries, in the opposite direction to the prior

After a large idiosyncratic **gain**, the stock underperforms equal weight. Every
one of the 20 cells in the up-arm grid is negative:

| z ≥ | h=5 | h=10 | h=20 | h=40 |
|---|---|---|---|---|
| 2.0 | −0.11% | −0.11% | −0.16% | −0.25% |
| **2.5** | −0.38% | **−0.60%** | −0.69% | −0.53% |
| 3.0 | −0.31% | −0.54% | −0.57% | −0.28% |
| 3.5 | −0.19% | −0.32% | −0.36% | −0.37% |
| 4.0 | −0.17% | −0.27% | −0.30% | −0.26% |

### What it survived in sample — and why that turned out to mean nothing

At z ≥ 2.5, h = 10 (n = 848, 40 stocks, ~86 fires/yr):

| test | result |
|---|---|
| pooled EV excess | **−0.601%**, t_naive −4.12 |
| block bootstrap (40 quarters) | t_block **−4.74**, p < 0.001, CI [−0.85%, −0.35%] |
| placebo, 400 reps, same stocks | random entries give −0.07%; observed is more negative than **400/400** draws |
| calendar-year stability | negative in **11 of 11** years (2016–2026) |
| cross-sectional stability | negative mean for **31 of 40** names |
| leave-one-stock-out | EV range [−0.65%, −0.55%] — no single-name dependence |
| leave-one-year-out | EV range [−0.66%, −0.52%] — no single-year dependence |
| data integrity | `log_return` matches `log(close).diff()` to 1e-15; no fire has \|raw move\| > 14.4%, so no split/dividend artifacts |
| accrual profile | gradual across days 1–10, not a day-1 jump — argues against a stale-price or microstructure artifact |

### What it is *not*

- **Not a generic vol-shock effect.** Pooling on \|z\| ≥ 2.5 gives −0.28% (t −2.54),
  roughly the average of a strong up arm and a null down arm. The effect is
  specific to up-shocks, and that asymmetry has no confirmed mechanism yet.
- **Not just 1-month reversal.** Fired names do have high trailing momentum
  (mom21 +5.2% vs universe +1.0%), but EV is negative in every mom21 quartile and
  is *strongest* in the lowest one (−0.99% vs −0.39% for the highest). It is not
  riding the short-term reversal factor.

### The weaknesses, stated plainly

1. **EV does not scale with shock size.** By strength quartile: −0.46% / −1.13% /
   −0.49% / −0.32%. The middle is the best and the largest shocks are the weakest.
   A real dislocation effect should strengthen with magnitude. This kills the
   confidence-ordering idea for this signal — every fire has to be sized the same,
   and it makes the z = 2.5 threshold look partly selected.
2. **This is the second hypothesis tested.** The up arm was run only after the
   down arm failed; across both arms that is ~50 grid cells and −0.60% is the best
   of them. The defence is that all 20 up-arm cells share the sign, which is far
   stronger than one good cell — but a fair point estimate is the grid average,
   roughly **−0.3% to −0.4%**, not −0.60%.
3. **No out-of-sample window.** Nothing here was held back from choosing z = 2.5
   and h = 10.

### Out-of-sample: the signal is dead

60 large caps outside the original 40 (`scratchpad/oos.py`), pulled over the same
2015-05-28 → 2026-07-20 window, with **every parameter frozen** at the in-sample
choice and the new universe supplying its own equal-weight market and benchmark.
No re-tuning of any kind.

| | in-sample 40 | out-of-sample 60 |
|---|---|---|
| n | 848 | 1251 |
| ev_excess | **−0.601%** | **+0.135%** |
| t_naive | −4.12 | +0.84 |
| t_block | −4.74 | +0.85 (p = 0.40) |
| placebo | beats 400/400 | p = 0.115 |
| up-arm grid cells negative | **20 / 20** | **5 / 20** |

The sign flips and significance vanishes. The grid flips with it: what was a
uniformly negative up arm in sample is 15/20 *positive* out of sample.

**It is not universe composition.** The OOS 60 is more growth/tech heavy, so the
set was split mechanically — median full-period annualised return, and median
realised volatility, no judgement calls, one shot:

| split | n | ev_excess | t_block |
|---|---|---|---|
| low-return half (mature, closest to the original 40) | 573 | −0.019% | −0.11 |
| high-return half (growth) | 622 | −0.042% | −0.14 |
| low-vol half | 571 | −0.121% | −0.70 |
| high-vol half | 622 | +0.137% | +0.45 |

Every cell is ~0. The mature, low-return half — the one that most resembles the
in-sample universe — gives −0.02%, not −0.60%. The effect is not present in any
subgroup of the new names.

**The only thing that replicates is the arm that was null.** Down-shock EV at
h = 10 is within a rounding error across both universes: z ≤ −2.5 gives −0.085%
in sample and −0.085% OOS; z ≤ −4.0 gives −0.231% and −0.222%. Consistent,
economically trivial, and insignificant in both — which is what a real, tiny
effect looks like, and is the shape the up arm should have had.

### The methodological lesson, which is the actual deliverable

The up arm passed **every** check available within its own sample: block
bootstrap over 40 quarters, a 400-rep placebo it beat unanimously, negative in
11/11 years, negative for 31/40 names, leave-one-stock-out range of only 10bp,
leave-one-year-out range of 15bp, and clean data integrity. All of it was
worthless, because all of it tests *robustness within the sample* and none of it
tests *generalisation to new assets*.

Standing rule from here: **a signal is not real until it has been run on tickers
that had no part in constructing it.** Held-out time windows are not enough —
the in-sample effect was stable across all 11 years, so any time-based split
would have confirmed it too. Only a fresh cross-section caught this.

Practical consequence: the OOS 60 should be reserved as a validation universe and
kept out of every future signal's construction and threshold choice.

---

## Regime conditioning — the "levels work in non-trending periods" assumption

Tested 2026-07-21, before building anything, because it is load-bearing for the
regime-conditioned technical-level idea and because level bounces have far too
few events to answer it. Instrument: generic cross-sectional short-horizon
reversal on the 40, thousands of dates. Pre-registered: efficiency ratio
`ER_20 = |Σr| / Σ|r|` over the trailing 20 days (near 1 = clean trend, near 0 =
chop), split at the **median**, no other cut examined. Test is the **paired**
per-date difference (low-ER half − high-ER half), Newey-West at lag h.

### The assumption is not supported, and the sign points the other way

Primary cell (ER cross-sectional, reversal k=5, target h=10):

| half | reversal IC | t_NW |
|---|---|---|
| low ER (choppy) | **+0.0016** | +0.17 |
| high ER (trending) | **+0.0136** | +1.41 |
| paired difference | **−0.0121** | −1.29 |

Reversal works *better* in the trending half. Not significant — but the sign is
negative in **8 of 8** cells across both ER conditioners (stock-level and
market-wide), at every k and h tried.

**The useful number is the interval, not the point estimate.** SE on the paired
difference is 0.0094, so the 95% CI is roughly [−0.031, **+0.006**]. Any
improvement larger than +0.006 IC from conditioning on chop is ruled out at 95%.
That is decisive for our purposes: even the optimistic end of the interval is too
small to rescue a strategy that starts at EV ≈ 0.

The literal "not trending *up*" reading (conditioner 3, signed 63-day momentum
rank) is a flat wash: differences +0.0066 / +0.0009 / −0.0004 / +0.0042, every
|t| < 0.7.

### A second, independent instrument agrees

Down-shock events (z ≤ −2.5, h=10, n=886) split by the same regimes — a
different mechanism from cross-sectional reversal, and closer to "price moved far
from a reference, does it come back":

| conditioner | low half EV | high half EV |
|---|---|---|
| ER cross-sectional | **−0.334%** | **+0.172%** |
| ER market-wide | −0.181% | +0.012% |
| mom63 cross-sectional | −0.037% | −0.137% |

Same direction: reversion trades do *worse* in chop, not better.

### Why this is the plausible answer, not a fluke

Low ER selects names whose price path is dominated by noise relative to
displacement. The intuition that "chop means mean reversion works" conflates
*prices oscillate* with *oscillations are predictable* — they are the same
observation only if the oscillation has structure. It appears it mostly doesn't.

### What this does not rule out

A different regime measure, a different window, or level bounces behaving
unlike generic reversal. But shopping for a conditioner that works is exactly the
failure mode this test existed to prevent, and the second instrument agreeing
makes the mechanism-specific escape less likely.

### The inversion — taken to the OOS 60, one pre-specified shot

Two predictions written down before running (`scratchpad/regime_oos.py`),
identical code path, only the universe changed.

**Primary — the conditioning effect: CONFIRMED.**

| | paired diff (low−high ER) | t_NW | IC low ER | IC high ER |
|---|---|---|---|---|
| in-sample 40 | −0.0121 | −1.29 | +0.0016 | +0.0136 |
| **OOS 60** | **−0.0102** | −1.10 | +0.0110 | **+0.0212** (t +2.35) |

Sign and magnitude both replicate. On the OOS 60 **all 12 factorial cells are
negative**, including the mom63 conditioner that was a wash in sample — the
market-wide split is the largest (IC +0.0015 choppy vs +0.0368 trending).

**Secondary — ER as a standalone signal: FAILED.** In-sample IC +0.0139
(t_NW +1.62); OOS **+0.0000, t_NW +0.00**, and 6 of 12 years negative. Dead. (The
script printed "CONFIRMED" on a naive `mean > 0` check — the test was badly
specified, the result is a failure. Noted so the mislabel isn't inherited.)

So: trending names *do not* outperform, but reversion *within* them works better.
It is a conditioning effect only.

### Where that leaves it

The user's original hypothesis is now doubly refuted — wrong in sample, and its
opposite replicates out of sample. That is as settled as anything in this project.

The surviving object is **short-horizon reversal restricted to high-ER
(trending) names**: OOS IC +0.0212, t_NW +2.35, versus +0.0136 in sample. That is
better than `composite_score` on the same universe (+0.0135). But hold it loosely:

- the conditioning effect itself is not significant in either universe
  (t ≈ −1.1, −1.3); only its *sign* replicates
- the two universes' IC series correlate 0.61 — one confirmation, not two
- IC +0.021 is still "marginal to good", not a large edge, and no trade rule, cost
  model, or event study has been run on it
- **the OOS 60 has now been spent on this question.** Further refinement of this
  idea has no clean validation set left. A third universe, a different period, or
  a different asset class would be needed before trusting a tuned version.

### As a trade rule it does not convert (`signals.reversal_in_trend`)

Pre-specified: `top_n=3`, 5-day reversal, ER top half, hold 10 days, 10bps, entry
at the next close, excess over equal weight. Controls fixed in advance.

| universe | arm | n (per yr) | ev_excess | t_block | placebo p |
|---|---|---|---|---|---|
| **40** | **high-ER (primary)** | 3088 (280) | **−0.0001** | −0.08 | 0.100 |
| **60** | **high-ER (primary)** | 3415 (310) | **+0.0017** | +1.88 | 0.013 |
| 40 | A: no ER filter | 2857 (259) | +0.0005 | +0.64 | |
| 60 | A: no ER filter | 3057 (277) | +0.0009 | +0.89 | |
| 40 | B: low-ER half | 3143 (285) | −0.0009 | −1.26 | |
| 60 | B: low-ER half | 3534 (320) | −0.0007 | −0.78 | |

**Zero on the universe the effect was found on**, and marginal on the other. Worse
still, on the 40 the ER filter is *worse than no filter at all* (−0.0001 vs
+0.0005), which directly contradicts the conditioning story at the trade level.
Only the OOS 60 shows the predicted ordering high > none > low.

The `top_n` optimum also disagrees between universes — best at 1 on the 40
(+0.0016), at 2 on the 60 (+0.0030), degrading to negative by 8 in both. Two
samples disagreeing on where the optimum sits is the signature of noise.

Costs matter at this scale: 10bps against a gross EV of 9–27bps is a third to a
half of the whole edge.

### Why IC +0.021 became EV ≈ 0 — worth internalising

Two distinct lessons, and the second is the more important one for the redesign.

**Cross-sectional IC lives in the whole ranking, not in the tail.** IC is a rank
correlation across all 40 or 60 names; a top-N rule only ever touches the extreme.
An IC of +0.02 is consistent with a well-ordered middle and a noisy tail, and that
appears to be what this is. **Never promote an IC result to a trade rule without
running it as one** — the conversion is not automatic and here it failed entirely.

**A cross-sectional ranking signal is not a sparse signal.** At 280–310 fires/yr
with 10-day holds this rule carries ~12 concurrent positions out of 60. That is a
continuously rebalanced tilt wearing an alert's clothing — the opposite of the
email-driven, act-by-hand design this whole direction exists to serve. Sparse
signals have to come from **event triggers** (a threshold being crossed, something
happening on a specific day), not from ranking the universe every morning.
`idio_shock` had the right shape and the wrong content; this has the right content
and the wrong shape.

---

## Signal 2 — post-earnings drift (`earnings.pead`). Refuted.

Tested 2026-07-21. Pre-specification written before any return was computed
(`scratchpad/PRESPEC.md`). **The OOS 60 was not touched** — the primary failed on
the 40, so per the pre-spec step 3 never ran. The validation universe is unspent.

### The data turned out to be the easy part

`yf.Ticker(t).get_earnings_dates(limit=100)` — `limit` is hard-capped at 100, but
100 rows reaches back to ~2002, not the two years feared. All 100 tickers
returned, zero failures: 9,638 raw rows, **4,396 in the price window**, median 44
events per ticker, 396–400/yr with no gaps, and `EPS Estimate` / `Reported EPS`
100% populated. These are true announcement dates; `quarterly_income_stmt` gives
only period-end dates and was discarded unused. Cached to `data/earnings/`.

**The BMO/AMC convention was derived, not assumed** — the timestamp hour recovers
it, and the price reaction confirms it (abnormal |return|, normalised to trailing
66d): after-close events peak at t+1 (3.89×) and before-open at t (2.96×), with
nothing on t−1. Getting this wrong would have put a third of events a day off.

### The primary test fails, with power

Top quintile of standardized earnings-day abnormal return, long, entry_lag=1,
h=20, 10bps, excess over equal weight. n=310 (~31/yr).

| | |
|---|---|
| ev_excess | **−0.29%** |
| t_block | **−0.72** (p = 0.47) |
| block 95% CI | **[−1.09%, +0.48%]** |
| placebo, 1000 reps | **p = 0.79** |

Wrong sign, and random entries on the same names beat it 79% of the time. Block
SE is 0.40%, so **any 20-day drift above +0.48% is excluded at 95%** — a true
+0.8% effect would have been caught. A powered negative, not a shrug.

### Why it is structural rather than marginal

- **The surprise measure demonstrably works.** Top-quintile events move 5.74×
  their normal daily range on the reaction day, and the abnormal-return z
  correlates with the *analyst* EPS surprise at Spearman +0.25. The information is
  being measured; the drift is absent.
- **The quintile ordering has the wrong shape.** At h=20: Q1..Q5 = −0.40% /
  +0.19% / **+0.33%** / −0.24% / −0.29%. An inverted U at every horizon — the
  no-surprise quarters do best and both tails lose. PEAD requires a monotone ramp.
  A single unlucky cell could be noise; a whole ordering with the wrong shape
  cannot.
- **All 8 horizon × arm cells are negative.** The largest (up, h=60, −1.05%) runs
  *opposite* to PEAD.
- **An independent surprise measure agrees.** The same rule off the analyst EPS
  surprise is a flat wash, max |t_block| = 1.39 across eight cells — so the
  abnormal-return proxy was not the problem.
- Across ~56 context cells, two came in under p=0.10 and **none under 0.05** —
  fewer than chance would produce. There is no hidden result in the grid.

### Controls

**Not momentum**: fired names carry mom63 +2.78% vs universe +3.16%, slightly
*below* average, and EV is negative in 3 of 4 momentum quartiles.

**Not earnings-specific — and this is the interesting part.** 36% of
`idio_shock` fires (303 of 848) land within ±3 days of an earnings reaction date.
The *non-earnings* shocks are at least as negative as the earnings ones at h=10
(−0.65%, t −3.91 vs −0.51%, t −2.16). **Corollary: `idio_shock` was substantially
an earnings signal that was never labelled as one** — worth remembering, though it
does not resurrect it, since `idio_shock` failed on the OOS 60 anyway.

### The one thread, deliberately not pulled

Rank correlation of surprise vs forward excess return across all 1,530 events
(higher power than a quintile cut) carries PEAD's sign in 6 of 8 cells: ρ = +0.033
(t +1.20) at h=10, +0.032 (t +1.38) at h=20. Nothing reaches t = 1.4 and the h=20
CI is [−0.013, +0.079]. Not worth pursuing, and `reversal_in_trend` already
established in this repo that IC ≈ +0.02 does not convert into a trade rule —
here it demonstrably doesn't, since the tail trade is negative.

### Verdict

**PEAD is not harvestable in 40 US mega-caps over 2016–2026.** This should not be
surprising: it was documented on small, thinly-covered, illiquid names and has
decayed since the 1990s. That this universe is a poor test bed for it is itself
the finding. The effect may well be real elsewhere.

**The durable output is the data, not the signal** — 4,396 verified announcement
dates with estimates and actuals across all 100 tickers, reusable as an
event-window veto (don't hold into a print), as a volatility feature, or for any
future event study.

---

## Next steps

1. **Make the OOS universe permanent and make it mandatory.** Fold the 60-ticker
   pull into the data tiers, keep it out of all signal construction, and run every
   future candidate through it before it gets written up as anything.
2. ~~Re-check `alpha.composite_score` on the OOS 60.~~ **Done — it passed.**
   Full-period IC +0.0135 on the 60 vs +0.0113 on the 40; post-2021 +0.0250 vs
   +0.0306. No sign flip, no ticker-selection pathology. But the same exercise
   found the headline +0.039 was inflated by 10-day target overlap; corrected,
   nothing reaches significance in either universe. Full table in
   `forecasting_notes.md`. Two methodological carry-overs:
   - **Newey-West (lag ≥ horizon) or every-h-th-date on all IC t-stats.** With a
     10-day target, ~2540 daily ICs are worth ~254 independent observations, and
     `sqrt(n_dates)` overstates the t by roughly 2.5×. This affects every IC
     quoted in this repo before 2026-07-21.
   - **The OOS universe controls for tickers, not for period.** The two
     universes' daily IC series correlate 0.61 — same market, same dates. A
     cross-sectional replication is one confirmation, not two, and it cannot
     clear a signal that is really a period effect.
3. **Signal 2.** The design only pays with several independent signals. Best
   remaining candidate is post-earnings drift (needs earnings dates), which has
   the outside prior and the event sparsity the design wants.
4. **The journal.** Log every fire with its EV estimate at fire time and fill in
   the realized excess at exit, with the signal and the acted-on decision as
   *separate* fields. Otherwise the record measures discretion, not the signal.

The premise of the redesign is still intact, and untested. Sparse conditioning
*could* recover edge that an all-days average drowns — this attempt just didn't
find any. What it did produce is a harness that killed a very convincing-looking
signal in about an hour, which is what it was built for.
