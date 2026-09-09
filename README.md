# trading-new

Research code for a two-stage equity system: a **forecasting** model whose
output feeds a **portfolio** model. Long-only over 40 large caps
(`config.ticker_list`), market exposure accepted.

## Notes files — read these first

| file | covers |
|---|---|
| `forecasting_notes.md` | pipeline architecture, target choice, volatility model (stage C), return model (stage A), evaluation conventions |
| `tech_levels_notes.md` | the technical support/resistance level work: causal construction, grid search, validation, and why it did not make it into the forecast |
| `signals_notes.md` | the sparse-signal redesign: event-study harness, the first signal and its refutation, and the out-of-sample rule that came out of it |

## Status (2026-07-21)

- **Stage C — volatility**: built. HAR+RF hybrid, R² 0.21 at a 10-day horizon.
- **Stage A — cross-sectional return**: built. A zero-parameter momentum
  composite, which beat every fitted model tried. Replicates on the OOS 60, but
  the original +0.039 was inflated by target overlap — **plan on IC ~ +0.015**,
  not significant in either universe once corrected. See `forecasting_notes.md`.
- **Stage B — market timing**: deliberately not attempted.
- **Stage D — volume**: built. R² 0.297 on the OOS 60 at h=10; genuinely distinct
  from volatility (only 24% forecast overlap). Not a trading signal.
- **High-vol regime detection**: cross-sectional works (OOS AUC 0.822), per-stock
  time-series does not (loses to its own baselines OOS).
- **Portfolio model**: deliberately dropped in favour of sparse signals.
- **Signals tested and refuted**: idiosyncratic shock, reversal-in-trend, PEAD.
  All four validation universes' worth of caveats live in `signals_notes.md`.
- **Technical levels**: validated as a bounce detector, but did not survive as a
  forecast input or as a trading overlay. AAPL is still hand-traded daily off
  `tech_level_live.py`, which is the honest out-of-sample test.

## Modules

**Forecasting**
- `forecast.py` — panel assembly (`build_panel`), pipeline, `NON_STATIONARY_COLS`
- `volatility.py` — forward realized-vol model, `fit_predict_sigma`
- `alpha.py` — stage-A features and `composite_score`

**Technical levels**
- `tech_levels.py` — core mechanics: `find_touches`, `build_levels_causal`, `score_touches`
- `tech_level_search.py` — grid-search driver (parallel across stocks)
- `tech_level_scoring.py` — the four hit-rate metrics from the search log
- `tech_level_input.py` — levels → per-date model features, `select_combos`
- `tech_level_trades.py` — event-level trade P&L
- `tech_level_continuation.py` — post-breakout forward returns
- `tech_level_live.py` — daily hand-trading report (`./venv/bin/python tech_level_live.py`)
- `tech_level_app.py` — Streamlit inspector

**Sparse signals**
- `events.py` — event-study harness: `full_report`, block bootstrap, placebo
- `signals.py` — signal registry; every signal returns `fires[stock, fire_date, side, strength]`
- `earnings.py` — announcement dates + the PEAD signal (refuted; the *data* is the keeper)

**Predictability studies** (what is forecastable, not what is tradeable)
- `vol_regime.py` — high-volatility periods as classification
- `volume_model.py` — volume level + high-volume periods

**Data / infra**
- `stock_class.py` — `Stock` / `StockList`, indicators, bronze→silver→gold tiers
- `config.py` — paths, `ticker_list`, and `oos_ticker_list` (validation universe)

## Conventions worth knowing before changing anything

- **A signal is not real until it runs on tickers that had no part in building
  it.** `config.oos_ticker_list` (60 names, disjoint from `ticker_list`) exists
  for this and must stay out of every construction and threshold choice. Held-out
  *time* windows are not a substitute — the one signal this caught was stable
  across all 11 years and still failed on a fresh cross-section.
- **Causality is checked by truncation**, not by inspection: recompute features
  on a truncated history and require the shared dates to match. This is what
  caught the prominence-scaling bug.
- **Purge ≥ horizon days** between train and test. Targets span multiple days
  and overlap; `TimeSeriesSplit` does not purge.
- **Cross-sectional IC** (within-date, across stocks) is the evaluation metric,
  not MAE and not pooled rank-IC.
- **No raw price or volume levels as features** — see `NON_STATIONARY_COLS`.
- Selection choices (which combo, which feature) must use only data before the
  panel starts, or the panel inherits the lookahead.
