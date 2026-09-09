# Forecasting pipeline — session notes

Companion to `tech_levels_notes.md`, which covers the technical-level work in
detail. This file covers the pipeline the levels were meant to feed: the
architecture, the volatility model (stage C), the return model (stage A), and
what actually survived validation.

**Status as of 2026-07-21.** Stage C and stage A are built and evaluated. The
portfolio model is not started. The technical-level features were tested and
did not earn a place in the panel — see `tech_levels_notes.md`.

---

## Architecture

Two models, deliberately separated: a **forecasting** model whose output is the
**portfolio** model's input. The portfolio step sees current holdings, a
forecast per stock, and costs, and decides what to trade.

### Why the interface has to be cardinal

The portfolio step compares expected gain from rebalancing against the cost of
trading. Both sides must be in the same units, so **a pure ranking cannot be
the interface**: "AAPL is ranked #1 of 40" is compatible with +7% expected and
with +0.2% expected — trade in the first case, hold in the second, and rank
cannot tell them apart.

That does not mean the *loss function* must be squared error. Two separable
choices, easy to conflate:

- train on returns, consume the ordering (sort the predictions) — rank
  robustness at consumption time is free
- train on ranks, recover magnitude by calibration (isotonic regression maps
  score → historical mean realized return, preserving the ordering)

### Three stages, split by what is actually predictable

| stage | target | realized | role |
|---|---|---|---|
| A. cross-sectional | relative return | IC ~0.04 | where the signal lives |
| B. market / common | universe average return | not attempted | market timing |
| C. volatility | realized vol | **R² 0.21** | sizing + target scaling |

Assembled as `E[r_i] = drift + spread_i × sigma_hat_i`.

**Stage B is deliberately not built.** Daily/weekly market direction is close to
unforecastable, and a noisy market forecast fed to the portfolio model would
churn the whole book. Constant drift for now; add timing only if it can be
demonstrated out of sample to the same standard as everything else here.

**Volatility is a separate model, not a second output head.** Vol is ~50x more
predictable than direction (R² 0.21 vs IC ~0.004). A multi-output tree chooses
splits to reduce the *sum* of variance across targets, so the easy target would
dominate every split and degrade the return forecast.

### Multi-output across stocks: considered and rejected

Forcing one model to predict all 40 tickers jointly is not a free efficiency
win. The econometric intuition (Zellner's SUR: joint estimation only helps when
equations have *different* regressors — identical regressors collapse to
equation-by-equation OLS) carries over to trees: sklearn's multi-output RF forces
every ticker to share the *same tree structure*, so a split that suits the
majority silently outvotes minority names. What captures cross-stock correlation
in a point forecast is **cross-sectional features**, not a multi-output head.
Pooled long panel (`n_dates × n_stocks` rows, stock as a feature) instead.

### Long-only, market exposure accepted

Consequences that shaped everything downstream:

1. **The benchmark is equal-weight buy-and-hold of the 40, not zero.** Most
   return and variance come from the market component we deliberately don't
   forecast.
2. **The top of the ranking matters, the bottom barely does.** Max expressible
   negative view is dropping to zero weight, ~−2.5% relative with 40 names; a
   positive view can be several times larger. Top-N precision maps to P&L;
   whole-cross-section rank correlation understates what matters.
3. **Turnover discipline matters more than in a long/short book** — every change
   of mind is a round trip. Forecast horizon should not be shorter than the
   holding period that costs allow.

---

## Target choice

**Settled, don't re-litigate:**

- **Log vs simple return at daily frequency: irrelevant.** Difference is
  second-order, far below the noise floor. Keep log (`calc_return` defaults to
  it). Convert at portfolio construction, where log returns don't aggregate
  across positions.
- **Scale by trailing volatility.** The highest-value target change, and it
  matters *because* the panel is pooled: squared-error loss on raw returns is
  dominated by whichever stocks are noisiest, so the model spends capacity on
  them and treats a large move in a calm stock as near-zero signal.
- **Horizon 10 days.** Daily returns are close to pure noise; signal-to-noise
  improves with horizon. 10d also scored best in the vol model, and matches the
  timescale of the technical-level signal. Cost: overlapping windows →
  autocorrelated residuals → **CV must purge ≥ horizon days**. `TimeSeriesSplit`
  does *not* do this; folds are adjacent.

**Chosen empirically:** absolute (not cross-sectionally demeaned) vol-scaled
10-day log return. Demeaning was tested and is supported via
`build_alpha_panel(demean_target=True)`.

---

## Stage C — volatility model (`volatility.py`)

Target: **log** realized vol over the next `horizon` days. Log because realized
vol is strongly right-skewed while log RV is near-Gaussian, which is what
squared-error loss assumes. Features: HAR cascade (trailing RV over 1/5/22/66
days), term-structure ratios, vol-of-vol, leverage-effect proxy, plus a
cross-sectional market vol factor.

### Results — walk-forward, 4 folds, purged by `horizon × 1.5` days

| horizon | random walk | HAR | RF | **HAR+RF hybrid** |
|---|---|---|---|---|
| 5d | 0.031 | 0.172 | 0.191 | **0.196** |
| **10d** | 0.043 | 0.188 | 0.209 | **0.211** |
| 20d | −0.042 | 0.178 | 0.192 | 0.190 |

**HAR is a hard baseline and RF's edge over it is modest** (+0.02 R²). Most of
the predictability is simple linear persistence — worth knowing before investing
further in the ML side.

### Why the shipped model is a hybrid

RF **loses to HAR in the 2020 vol-spike fold** at every horizon (0.229 vs 0.270
at 10d). A tree cannot predict outside the range of targets it saw in training,
so a pure RF systematically under-forecasts spikes larger than anything in its
history — precisely the regime where the forecast matters most for sizing.

The hybrid has HAR carry the level (linear, extrapolates) and RF fit only the
residual. Recovers most of the crisis-fold gap (0.229 → 0.245) while keeping
RF's advantage elsewhere. `fit_predict_sigma` uses it.

### Retransformation correction

The model fits in logs, so `exp(prediction)` recovers the conditional *median*,
not the mean — first check showed 22.2% predicted vs 24.7% actual. Added the
standard smearing correction `exp(s²/2)`: now 23.3% vs 24.8%, medians 21.6% vs
22.0%. Harmless when sigma_hat only rescales a target (a constant factor
cancels) but it would hand portfolio construction a systematically low risk
estimate. Residual gap is in the right tail — the same extrapolation limit.

---

## Stage A — cross-sectional return model (`alpha.py`)

### The result: the model is not a random forest

Adding the missing cross-sectional features made the RF **worse**:

| feature set | mean IC |
|---|---|
| own-stock only (rsi/sma_cross/variance) | −0.006 |
| + momentum / reversal / vol / beta | −0.009 |
| + cross-sectional ranks, market, dispersion | −0.010 |

Both model families then improved **monotonically** as capacity was removed —
RF −0.016 (leaf 10) → +0.008 (leaf 1000); Ridge −0.005 (alpha 1) → +0.010
(alpha 1e4). "Less model is better, all the way to the edge of the grid" is not
a tuning result; it says the panel supports far less capacity than a forest has.

### Why — the raw single-feature ceiling

Cross-sectional IC per feature, no model, no fitting:

| feature | IC | t |
|---|---|---|
| `a_mom_252` (12-month momentum) | **+0.031** | 4.60 |
| `a_mom_12_1` | +0.024 | 3.43 |
| `rsi_14` | +0.019 | 2.88 |
| `rsi_100` | −0.017 | −2.29 |
| `a_mom_63` (3-month) | −0.015 | −2.12 |

Textbook signs — 12-month momentum positive, intermediate-horizon momentum
reversing. But **max |IC| in the whole feature set is 0.031**, so there was never
enough for a flexible learner to find without fitting noise.

### The shipped model: `alpha.composite_score`

Zero parameters. Long-term momentum rank minus 3-month momentum rank, signs
taken from the literature, nothing estimated.

| model | fold ICs | mean IC | top-8 spread |
|---|---|---|---|
| RF (best) | | +0.008 | −0.003 |
| Ridge (best) | | +0.010 | +0.012 |
| **composite** | +0.048 / +0.027 / +0.032 / +0.051 | **+0.039** | **+0.055** |

Four times the best fitted model, positive in all four folds. The legs are
genuinely complementary, not merely averaged: momentum alone is negative in fold
1, reversal alone is negative in fold 4, the combination is positive throughout.

**Caveat:** the two legs were chosen after inspecting full-sample single-feature
ICs, which included the evaluation windows. That selection is not free. Defence
is that both are among the most replicated effects in the cross-sectional equity
literature and the signs came from there rather than being fitted.

### Revision, 2026-07-21 — the +0.039 does not survive contact

Two things were wrong with it. Details and method in `signals_notes.md`.

**The t-stats were inflated by target overlap.** A 10-day target makes
consecutive daily ICs share 9 of 10 days of outcome, so `mean/std*sqrt(n_dates)`
across ~2540 dates is not a valid t. Newey-West at lag 10 (or keeping every 10th
date, which leaves ~254 independent observations) gives:

| universe | period | IC | t_naive | **t_NW** | t_indep | years > 0 |
|---|---|---|---|---|---|---|
| in-sample 40 | full | +0.0113 | +2.31 | **+0.89** | +1.30 | 5/11 |
| **OOS 60** | full | +0.0135 | +3.30 | **+1.30** | +1.18 | 7/11 |
| in-sample 40 | post-2021 | +0.0306 | +4.69 | **+1.85** | +1.52 | 5/6 |
| **OOS 60** | post-2021 | +0.0250 | +4.44 | **+1.75** | +0.97 | 5/6 |

Nothing reaches conventional significance. The partial 2026 year does a lot of
work: excluding it, post-2021 falls to +0.0205 (40) and +0.0202 (60).

**But it did replicate on a fresh cross-section**, which is the test that killed
`signals.idio_shock`. On the 60 tickers of `config.oos_ticker_list`, which had no
part in choosing the legs, sign and magnitude both hold. The composite does not
have the ticker-selection pathology.

Caveat on that: the two universes' daily IC series correlate **0.61** — same
market, same dates. The OOS test controls for ticker selection, not for period,
so it is one confirmation, not two.

Which leg carries is not stable either: mom252 dominates on the 40 (+0.0139 vs
+0.0061), the reversal leg dominates on the 60 (+0.0132 vs +0.0068). Both legs
are positive in both universes — that is the argument for keeping the pair rather
than picking the apparent winner.

**Verdict: not refuted, not demonstrated.** It stays the stage-A model because it
is zero-parameter and has an outside prior, not because of its t-stat. **Plan on
IC ~ +0.015.**

---

## Stage C addendum — predicting *high-volatility periods* (`vol_regime.py`)

Added 2026-07-21. A different question from the R² above, and the distinction is
the whole point: **R² = 0.211 on log RV does not mean you can call turbulent
periods.** Level accuracy on a near-Gaussian, strongly persistent target is earned
in the dense middle; "will the next 10 days be turbulent" is tail detection
against a ~20% base rate.

Two labels, because they are different questions and they **disagree**:
*cross-sectional* (top quintile of the universe that date) and *time-series* (top
quintile of that stock's own completed history). Time-series thresholds are
**expanding and lagged by h rows** so only finished windows count — a full-sample
quintile would leak March 2020 back into 2016. Verified by truncation: recomputed
on history cut at 2022-01-03, all 53,520 shared rows match, max abs difference
0.0, zero label mismatches.

Baselines first, always: persistence (was the just-finished window top-quintile),
HAR-threshold, HAR-logit, then the RF.

### Fold-mean ROC-AUC, h=10

| label | universe | persistence | HAR-logit | RF | RF − pers |
|---|---|---|---|---|---|
| cross-sectional | 40 | 0.680 | 0.704 | **0.729** | +0.049 |
| cross-sectional | **60 (OOS)** | 0.771 | 0.794 | **0.822** | +0.051 |
| time-series | 40 | 0.659 | 0.669 | 0.674 | +0.015 |
| time-series | **60 (OOS)** | 0.668 | **0.682** | 0.662 | **−0.006** |

**Cross-sectional replicates; time-series does not.** On the 60 the time-series RF
*loses* to both its baselines, and its margin over persistence across h=5/10/20
averages +0.005 with a sign flip. The +0.015 on the 40 was noise.

Top-decile precision, cross-sectional h=10, off a 20% base: 55.9% (40) and 76.2%
(60), against persistence's 43.3% / 61.7%.

### The finding that matters: it is mostly a state lookup

Split by current state (cross-sectional, h=10):

| subset | n | base | pers AUC | RF AUC | RF p@10% |
|---|---|---|---|---|---|
| all | 86,880 | 0.200 | 0.680 | 0.727 | 0.558 |
| **calm now → transition** | 69,504 | 0.157 | 0.628 | 0.673 | 0.350 (2.23×) |
| turbulent now → persistence | 17,376 | 0.373 | 0.583 | 0.749 | 0.764 |

The base rate is 2.4× higher in the turbulent-now group, so most of the pooled
AUC is separating those two groups — which needs only the current state. **If you
already track current RV you hold ~90% of the available signal.** Transitions are
the hard half: the best case is ~2.2–3.0× lift off a 12–16% base, meaning **~35%
of flagged calm names turn turbulent and 65% do not.**

### Calibration — the honest caveat

Pooled calibration looks respectable only because already-turbulent rows dominate,
where the answer is easy. Restricted to currently-calm rows — exactly where the
model would be useful — the RF is **materially overconfident**: predicted 0.442 →
actual 0.325, 0.541 → 0.332, 0.638 → **0.361**. A calm stock flagged at 60% turns
turbulent about a third of the time. Same shape on the 60. On the time-series
label the cheap logistic is *better calibrated* than the forest.

Longer horizons score higher and mean less — AUC rises with h because averaging
more days makes the target more persistent, while the transition subsample barely
improves.

### Verdict

- **"Which names will be wild": moderately well, and better than free.** Margin
  survives a fresh cross-section, concentrated in transitions, calibrated below
  0.8. Usable as a risk-*ranking* input.
- **"Is this name about to get turbulent": no.** Use persistence or the HAR-logit
  — both free, and the logit is better calibrated.
- Reproduce: `PYTHONPATH=. ./venv/bin/python scratchpad/run_regime.py {40|60} 5,10,20`.

---

## Stage D — volume prediction (`volume_model.py`)

Added 2026-07-21. A predictability question, not a signal: nothing here says how
to trade. **Volume is about as predictable as volatility and an order of magnitude
better than returns** — R² 0.284 at h=10 on the 40, **0.297 on the fresh 60**.

Target and normaliser are the whole game. Never raw share volume (AAPL's mean log
volume falls 19.0 → 17.7 over the sample — a stock-identity label plus a clock,
the `NON_STATIONARY_COLS` failure mode). Instead
`log(volume) − log(median(volume, 66d))`, with a **rolling median not mean**,
because one earnings print at 3–8× normal would drag a 66-day mean for a quarter
and make the normaliser a lagged copy of the spikes being predicted. **The
target's normaliser is frozen at t**, or most of the target hides in the
denominator. Truncation-verified at 2022-06-30: all 32 features, target and
expanding label reproduce bit-for-bit on 65,839 shared rows, zero disagreements.

### R², 4 purged folds — baselines first (40 / 60)

| model | h=5 | h=10 | h=20 |
|---|---|---|---|
| random walk, unshrunk (0 params) | **−0.028** | **−0.118** | **−0.186** |
| persistence, one fitted slope | 0.210 | 0.147 | 0.088 |
| calendar only | 0.084 | 0.057 | 0.031 |
| HAR cascade (4 params) | 0.268 | 0.210 | 0.158 |
| HAR + calendar (15 params) | 0.343 | 0.271 | **0.194** |
| realized vol only | 0.094 | 0.081 | 0.069 |
| **RF, all features** | **0.363** / 0.362 | **0.284** / 0.297 | 0.189 / 0.221 |

**The unshrunk random walk scores negative R² at every h ≥ 5** — normalised volume
mean-reverts, and shrinkage is the single most important modelling act. One fitted
slope moves h=10 from −0.118 to +0.147.

**Don't buy the forest.** HAR+calendar is within 0.02 R² everywhere at 15
parameters, and at h=20 on the 40 it *beats* the RF (0.194 vs 0.189).

**Regime caveat — the recent fold is the weak one.** RF at h=10 by fold: 0.22 /
0.42 / 0.30 / **0.20**. Covid flatters every average. **Plan on R² ≈ 0.20.**

### Both tidy stories are refuted

**Not the calendar.** Calendar-only R² is 0.057 at h=10 and adds ~0.05 on top of
the volume model (~18% of it); volume's own history is worth ~0.18, four times as
much. Triple witching dominates the calendar coefficients (+0.64 at h=1, 3× the
next effect), and two of six real effects are *negative* — year-end and
holiday-adjacent lulls. Day-of-week barely registers.

**Not volatility in disguise.** Only **24%** of the full forecast's variance is
explained by a vol-only forecast, and trailing RV adds **+0.01 R²** on top of the
volume cascade. Contemporaneous correlation is high; *forecast* overlap is not.
Attribution: volume history ~0.18, calendar ~0.05, realized vol ~0.01. This is
genuinely new information relative to `volatility.py`.

### Classification, base rate 0.20, top decile

| label | h | calendar only | persistence | **RF** |
|---|---|---|---|---|
| cross-sectional | 10 | 1.11× (AUC 0.509) | 1.99× | **2.45×** (AUC 0.689 / 0.696) |
| own-history | 10 | 1.55× (AUC 0.621) | 2.24× | **2.56×** (AUC 0.717 / 0.739) |

**The calendar has a split personality, which is the most interesting result
here:** cross-sectionally it is a coin flip (AUC 0.501–0.510 at every horizon), on
the own-history label it is clearly useful (0.60–0.63). Scheduled market-wide
events tell you *when* the tape is busy, never *which stock*.

Calibration is good for the cross-sectional label at h=10 (Brier 0.145 vs 0.160)
and **bad for own-history at h=20**, where the curve inverts above 0.7 (predicted
0.84 → realised 0.39). That is expanding-threshold drift on a trending series.
**Use the cross-sectional label when you need calibrated probabilities.**

### Transition vs persistence — the prior expectation was refuted here

Unlike `vol_regime.py`, the skill is *not* mostly a state lookup. Absolute
precision nearly halves on currently-normal stocks (0.70 → 0.37 at h=10), but
almost all of that is the base rate falling 0.30 → 0.17. **Relative lift holds at
~2.1–2.3× on stocks that are not currently busy** — and on the own-history label
at h=10 the lift is *identical* in both subsets (2.29 vs 2.29). Replicates on the
60 (2.18 / 2.51).

### OOS — every finding transfers, including the negatives

R², AUC, lift, the calendar's cross-sectional uselessness, and the
currently-normal lift all reproduce on the 60. Unusual for this repo. Volume is a
mechanical microstructure quantity rather than an alpha, and lacks the fragility
that killed `signals.idio_shock`.

### Verdict and next steps

Residual std of forward 10-day log volume is 0.217 vs 0.256 unconditional — a
forecast typically within ±24% (1σ) vs ±29% from a constant. Useful, unspectacular.
The honest margin is the gap from free persistence (~2.0×) to the model (~2.5×).

1. **Earnings dates are the highest-value addition** — the spike proxy is
   measurably worthless (coefficient −0.009). **These now exist**: `earnings.py`
   and `data/earnings/` hold 4,396 verified announcement dates across all 100
   tickers, fetched the same day. This is the obvious next iteration.
2. Fit per-horizon shrinkage explicitly instead of letting the RF discover it.
3. Do not invest in nonlinearity — HAR+calendar is within 0.02 R² at a fraction
   of the complexity.

---

## Panel infrastructure (`forecast.py`)

- `create_long_x_y(..., horizon=h)` — target is cumulative log return over
  t+1..t+h. `horizon=1` reproduces the original next-day behaviour.
- `build_panel(horizon=10, vol_scale=True)` — attaches level features, assembles
  the long panel, divides the target by `sigma_hat × sqrt(horizon)`, truncates
  to start strictly after `selection_cutoff`.
- `build_pipeline(max_features=0.5)` — sklearn's regressor default is 1.0 (every
  split sees every feature), which lets dominant features crowd out weaker ones.
- **`NON_STATIONARY_COLS`** — raw price and volume *levels* (`close`, `high`,
  `low`, `open`, `volume`, `sma_9`–`sma_100`) are excluded by default. Trees
  split on absolute values, so in a pooled panel these act as stock-identity and
  regime labels that fit the training window and generalize to nothing. The
  baseline was anti-predictive (mean IC −0.018, negative in all four folds)
  until they were dropped.
- `NON_FEATURE_COLS` — the gold tier carries duplicated date columns (`dt`,
  `dt.1`–`dt.3`) and a redundant `ticker`; leaving them in makes the panel
  non-numeric and `RandomForest.fit` raises.

### Evaluation conventions

- **Cross-sectional IC** — within each date, Spearman(predictions, realized)
  across the 40 stocks, averaged over dates. This is the decision-relevant
  metric for long-only selection. *Pooled* rank-IC across all rows conflates
  cross-sectional skill with period effects and flatters the model — don't use it.
- Rough calibration for equities: 0.01–0.02 marginal, 0.03–0.05 genuinely good,
  >0.10 almost always a bug or a leak.
- **Purge ≥ horizon days** between train and test, always.
- MAE is the wrong metric here — it barely moved (0.01318 → 0.01307) while IC
  changed 10x, because it measures error magnitude and the decision consumes
  ordering.

---

## What survived, and what didn't

**Survived:** a two-term momentum composite (stage A) and a HAR-based volatility
model (stage C). Both simple; both with priors from outside this dataset.

**Did not survive:** the technical-level features (see `tech_levels_notes.md` —
the apparent IC was a time-proxy artifact), the random forest as the stage-A
model, and the level-based portfolio overlay (zero pooled EV across the credible
names).

---

## Next steps

1. **Portfolio model.** Everything it needs now exists: `composite_score` for
   expected relative score, `fit_predict_sigma` for risk, and equal-weight
   buy-and-hold as the benchmark. Cost-aware, long-only, with a no-trade band
   widened by forecast uncertainty — RF cross-tree dispersion was the original
   idea for that, but the shipped stage-A model has no trees, so uncertainty
   needs another source (cross-sectional dispersion of the composite, or the
   spread between its two legs).
2. **Validate the stage-A composite on a clean window** that had no part in
   choosing its two legs — the honest fix for the selection caveat.
3. **Stock characteristics instead of one-hot stock identity.** One-hot lets a
   tree ask only "is this exactly ticker X"; it cannot express "all high-beta
   tech names" in one split, and says nothing about an unseen ticker.
   `a_beta` exists; sector is unavailable (`set_sector` returns Unknown for the
   whole universe).
4. **Stage B**, only if market timing can be demonstrated out of sample.
