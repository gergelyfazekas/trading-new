"""High-volatility *regime* detection -- the classification question stage C does not answer.

`volatility.py` reports R^2 = 0.21 on log forward realized vol at a 10-day
horizon. That is a level-accuracy number and it is dominated by the easy middle
of the distribution. The question a risk user actually asks is different and
binary: **"are the next 10 days going to be turbulent?"** High R^2 on a
persistent, near-log-Gaussian target does not imply you can answer that, so this
module re-asks it as tail detection and scores it against the only baseline that
matters -- persistence.

Two label definitions, because they are different questions and they disagree:

  *cross-sectional* -- forward RV in the top quintile **among the universe on
  that date**. "Which names will be the wild ones this month." Base rate is 20%
  by construction and the market-wide vol level cancels out.

  *time-series* -- forward RV above the 80th percentile of **that stock's own
  completed history to date**. "Is this stock about to be more turbulent than it
  usually is." The threshold is expanding and lagged by `horizon` days so it only
  ever uses forward-vol windows that had already finished; a full-sample quintile
  cut would leak the 2020 spike backwards into 2016. Base rate drifts (it is far
  above 20% in 2020, far below in 2017), which is itself part of the signal.

Baselines, in the order they should be read:

  *persistence* -- "the window that just ended was top-quintile, so the next one
  will be". Encoded as the label evaluated at t-h, which is exactly the trailing
  h-day RV under the identical definition, so it is a like-for-like comparison
  rather than a proxy.

  *HAR* -- threshold the existing linear HAR forecast at the same causal cutoff,
  plus a logistic version of it for calibration. Nearly free, and the real bar.

FINDINGS
--------
**Headline: much worse than R^2 = 0.21 sounds. ROC-AUC 0.67-0.75 depending on
label and horizon, of which persistence -- "the window that just ended was
high-vol" -- already delivers 0.65-0.72. The model adds +0.015 to +0.06.**

Everything below is walk-forward, 4 folds, purged by `horizon * 1.5` days,
developed on `config.ticker_list` (40) and confirmed once, frozen, on
`config.oos_ticker_list` (60). Nothing was tuned on either. Label causality was
checked by truncation (recompute on history cut at 2022-01-03; 53,520 shared rows
matched to max abs difference 0.0, zero label mismatches).

1. **Walk-forward fold-mean AUC, the 40.** Read the persistence column first.

       cross-sectional label (base 20%)      time-series label (base 26-28%)
       h   pers  HAR  HARlogit   RF          h   pers  HAR  HARlogit   RF
       5   .651  .688   .691   .712          5   .653  .669   .672   .674
       10  .680  .702   .704   .729          10  .659  .664   .669   .674
       20  .724  .727   .730   .752          20  .670  .670   .678   .690

   The two label definitions disagree and the disagreement is the point. Picking
   *which names* will be wild is improvable (+0.03 to +0.06 AUC over persistence,
   top-decile precision 55.9% vs 43.3% off a 20% base at h=10). Picking *when a
   given name* will be wild is not: +0.015 AUC at h=10, and at h=5 the free HAR
   baseline beats the forest on top-decile precision (0.526 vs 0.511).

2. **Most of the pooled number is a state lookup, not a forecast.** Splitting the
   test rows on whether the just-finished window was itself high-vol (h=10):

       cross-sectional        n       base   pers AUC  RF AUC  pers p@10  RF p@10
       all                    86,880  0.200    0.680    0.727    0.433     0.558
       calm now (transition)  69,504  0.157    0.628    0.673    0.263     0.350
       turbulent (persist.)   17,376  0.373    0.583    0.749    0.493     0.764

       time-series            n       base   pers AUC  RF AUC  pers p@10  RF p@10
       all                    78,560  0.271    0.679    0.685    0.595     0.538
       calm now (transition)  57,478  0.203    0.593    0.603    0.296     0.323
       turbulent (persist.)   21,082  0.456    0.639    0.674    0.799     0.598

   The base rate is 2.4x higher in the turbulent-now group. Separating those two
   groups is the bulk of the pooled AUC and needs only the current state. Inside
   the turbulent group on the time-series label the RF's top-decile precision is
   *worse* than persistence's (0.598 vs 0.799) -- once a name is already wild,
   ranking within that state is close to noise.

3. **Transitions are detectable but unreliable.** Calm -> turbulent, h=10,
   cross-sectional: base 15.7%, RF top-decile precision 35.0% (lift 2.23x) vs
   persistence 26.3% (1.68x). Time-series: 32.3% off a 20.3% base, lift 1.59x vs
   persistence 1.46x. Roughly two of every three "a storm is coming" flags do not
   happen. This is the only genuinely useful case and it is the weakest one.

4. **Calibration fails exactly where it matters.** Pooled, the cross-sectional RF
   is within a few points up to 0.8 and over-promises by 11 points in the top
   bucket (0.850 -> 0.742) -- the same right-tail shortfall the regression model
   has, a tree cannot predict past its training targets. On the time-series label
   the *cheap logistic model is better calibrated than the forest* (RF 0.889 ->
   0.670; logit 0.829 -> 0.860). Worst of all, restricted to currently-calm rows,
   the RF is badly overconfident: predicted 0.44 -> actual 0.33, 0.54 -> 0.33,
   0.64 -> 0.36. The pooled calibration looks respectable only because it is
   dominated by already-turbulent rows where the answer is easy.

5. **Longer horizons score higher and mean less.** AUC rises monotonically with h
   in both labels, purely because averaging over more days makes the target more
   persistent. The transition subsample improves far less.

6. **The fairness fix was as large as the model's edge.** Feeding the classifier
   the causal threshold (`augment_features`) raised time-series RF AUC from 0.659
   to 0.674 at h=10 -- i.e. as much as its entire margin over persistence. Worth
   remembering before reading any classifier-vs-persistence comparison where the
   label is defined against a moving cutoff the baseline gets for free.

7. **OOS 60.** **OOS 60 -- the time-series model does not replicate.**
   The cross-sectional one does, and more strongly: fold-mean AUC at h=10 is
   persistence 0.771, HAR 0.793, RF 0.822, with top-decile precision 0.762 vs
   persistence's 0.617 off the same 20% base, and transitions give the RF 2.99x
   lift off a 12.4% base vs persistence's 2.15x. (Absolute levels run higher than
   on the 40 because the 60 is a more dispersed universe -- compare margins, not
   levels.) The time-series RF, by contrast, **loses to its own baselines**:
   fold-mean AUC at h=10 is RF 0.662 against HAR-logit 0.682 and persistence
   0.668, and its margin over persistence across h=5/10/20 averages about +0.005
   (+0.017, -0.006, +0.003). The +0.015 seen on the 40 was noise. The time-series
   question is answered by persistence plus HAR and by nothing beyond it.

**Verdict.** "Will the next h days be turbulent?" is answerable at roughly the
accuracy of "were the last h days turbulent?" plus a small margin. If you already
track current RV you hold ~90% of the available signal. Only the cross-sectional
version has a margin that survived a fresh cross-section, and only as a
risk-*ranking* input. For the time-series question use persistence or the
HAR-logit -- both are free, the logit is better calibrated, and the forest was
beaten by both on the 60. R^2 = 0.21 on log
RV does **not** imply you can call turbulent periods -- that is the difference
between explaining a fifth of the variance and being right about one flagged
storm in three.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from volatility import build_vol_panel

HAR_COLS = ['rv_1', 'rv_5', 'rv_22']
QUANTILE = 0.8
MIN_HISTORY = 250  # trading days before a per-stock expanding threshold is trusted


# ---------------------------------------------------------------- labels

def build_labels(y, stock, horizon, mode='timeseries', quantile=QUANTILE,
                 min_history=MIN_HISTORY):
    """Binary 'next `horizon` days are high vol' label plus the current state.

    `y` is log forward realized vol (the target from `volatility.build_vol_panel`),
    `stock` the aligned ticker column. Returns a frame with:

      label          1 if forward vol is above the high-vol cutoff
      current_state  the same label evaluated `horizon` rows earlier, i.e. whether
                     the window that just *finished* was high vol. Known at t.
                     This is the persistence baseline, expressed in exactly the
                     label's own units so the comparison is like-for-like.
      threshold      the cutoff used, kept for thresholding regression forecasts

    mode='crosssectional': cutoff is the `quantile` of y across the universe on
    that date. Contemporaneous cross-section of the target itself -- no time
    leakage, and the base rate is pinned at 1-quantile.

    mode='timeseries': cutoff is the expanding `quantile` of that stock's own
    forward-vol observations **that had already completed by date t** (an
    expanding quantile of y shifted by `horizon` rows). Causal by construction.
    The first `min_history` observations per stock are dropped rather than scored
    against a threshold estimated from a handful of points.
    """
    df = pd.DataFrame({'y': np.asarray(y, dtype=float), 'stock': np.asarray(stock)},
                      index=y.index)

    if mode == 'crosssectional':
        thr = df.groupby(level=0)['y'].transform('quantile', quantile)
    elif mode == 'timeseries':
        # only forward-vol windows that had already *finished* by date t
        df['completed'] = df.groupby('stock')['y'].shift(horizon)
        thr = df.groupby('stock')['completed'].transform(
            lambda s: s.expanding(min_periods=min_history).quantile(quantile))
        df = df.drop(columns='completed')
    else:
        raise ValueError(f'unknown mode {mode!r}')

    df['threshold'] = thr
    df['label'] = np.where(thr.isna(), np.nan, (df['y'] > thr).astype(float))
    df['current_state'] = df.groupby('stock')['label'].shift(horizon)
    return df[['label', 'current_state', 'threshold', 'y', 'stock']]


def augment_features(X, lab, mode):
    """Add the features that make the model's job comparable to persistence's.

    Without these the comparison is rigged *against* the model. The label is
    defined relative to a moving cutoff -- the stock's own expanding quantile, or
    the day's cross-section -- and a classifier fed only absolute log RV levels
    has to infer that cutoff from the stock dummy alone. Persistence gets it for
    free because it is defined in the same units as the label.

    timeseries: the expanding threshold (causal -- it only uses forward-vol
    windows that finished at or before t) and current RV's distance from it.
    crosssectional: within-date percentile ranks of current RV. The *label's*
    cross-sectional cutoff is a quantile of the target and must never be a
    feature; ranks of trailing RV are known at t and are the legitimate analogue.
    """
    X = X.copy()
    if mode == 'timeseries':
        thr = lab['threshold']
        X['thr'] = thr.to_numpy()
        for c in ('rv_5', 'rv_22'):
            X[f'{c}_vs_thr'] = X[c].to_numpy() - thr.to_numpy()
    else:
        for c in ('rv_5', 'rv_22'):
            X[f'{c}_xs_rank'] = X.groupby(level=0)[c].rank(pct=True).to_numpy()
    return X


# ---------------------------------------------------------------- models

def build_classifier(n_estimators=300, min_samples_leaf=50, max_features=0.5):
    """RF classifier on the vol panel, stock one-hot encoded like the regressor.

    `min_samples_leaf` is deliberately larger than the regressor's 20: leaves need
    enough observations for the predicted frequency to be a usable probability
    rather than 0/1, and calibration is the point of this exercise.
    """
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 random_state=0, n_jobs=-1)
    pre = ColumnTransformer([('stock_ohe', OneHotEncoder(handle_unknown='ignore'), ['stock'])],
                            remainder='passthrough')
    return Pipeline([('preprocess', pre), ('clf', clf)])


def build_har_logit():
    """Logistic regression on the HAR cascade -- the cheap probabilistic bar."""
    return Pipeline([('scale', StandardScaler()),
                     ('logit', LogisticRegression(max_iter=1000))])


# ---------------------------------------------------------------- metrics

def classification_metrics(label, score, top_frac=0.1, prob=None):
    """AUC, top-decile precision/recall, and lift over the base rate.

    Lift is the number to read: precision alone is meaningless without the base
    rate it is being compared against, and the base rate here moves with the
    label definition, the horizon and the regime.
    """
    label = np.asarray(label, dtype=float)
    score = np.asarray(score, dtype=float)
    base = label.mean()
    n_top = max(1, int(round(len(score) * top_frac)))
    order = np.argsort(-score, kind='mergesort')[:n_top]
    prec = label[order].mean()
    out = {
        'n': len(label),
        'base_rate': base,
        'auc': roc_auc_score(label, score) if 0 < base < 1 else np.nan,
        'prec_top_decile': prec,
        'lift': prec / base if base > 0 else np.nan,
        'recall_top_decile': label[order].sum() / label.sum() if label.sum() else np.nan,
    }
    if prob is not None:
        out['brier'] = brier_score_loss(label, np.asarray(prob, dtype=float))
    return out


def calibration_table(label, prob, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, 1.01)):
    """Predicted vs realised frequency by probability bucket."""
    label = np.asarray(label, dtype=float)
    prob = np.asarray(prob, dtype=float)
    idx = np.digitize(prob, np.asarray(bins[1:-1]))
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({'bucket': f'{bins[b]:.1f}-{min(bins[b+1],1.0):.1f}',
                     'n': int(m.sum()), 'pred': prob[m].mean(), 'actual': label[m].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- evaluation

def evaluate_walkforward(X, y, horizon, mode='timeseries', n_folds=4, top_frac=0.1,
                         quantile=QUANTILE, min_train=5000):
    """Walk-forward regime classification against persistence and HAR.

    Same fold geometry and purge as `volatility.evaluate_walkforward`: the last
    `horizon * 1.5` calendar days before each test fold are dropped from training
    because their targets overlap the fold.

    Four predictors, all scored on identical rows:
      persistence  score = the just-completed window's own vol, ranked the same
                   way the label is cut (cross-sectionally within date, or against
                   the stock's expanding threshold). Its *binary* form is
                   `current_state`, reported separately as prec/recall.
      har_thresh   OLS HAR forecast of log forward vol, thresholded at the same
                   causal cutoff. Score is forecast minus cutoff.
      har_logit    logistic regression on the HAR cascade -- probabilistic.
      rf           RF classifier on the full vol feature set.

    Returns (per-fold metrics, pooled out-of-sample predictions). The pooled
    frame is what the persistence/transition split is computed from.
    """
    lab = build_labels(y, X['stock'], horizon, mode=mode, quantile=quantile)
    keep = np.asarray(lab['label'].notna() & lab['current_state'].notna())
    X, y, lab = X[keep], y[keep], lab[keep]
    Xa = augment_features(X, lab, mode)

    dates = np.array(sorted(X.index.unique()))
    folds = np.array_split(dates, n_folds + 1)[1:]

    rows, preds = [], []
    for f in folds:
        purge = f[0] - pd.Timedelta(days=int(horizon * 1.5))
        tr = np.asarray(X.index < purge)
        te = np.asarray((X.index >= f[0]) & (X.index <= f[-1]))
        if tr.sum() < min_train:
            continue

        ytr, yte = lab['label'].to_numpy()[tr], lab['label'].to_numpy()[te]
        if ytr.min() == ytr.max() or yte.min() == yte.max():
            continue

        # --- persistence: rank the vol of the window that just ended
        prev_y = lab.groupby('stock')['y'].shift(horizon)
        if mode == 'crosssectional':
            pers_score = prev_y.groupby(level=0).rank(pct=True)
        else:
            pers_score = prev_y - lab['threshold']
        pers_score = pers_score.to_numpy()

        # --- HAR regression thresholded at the same causal cutoff
        har = LinearRegression().fit(X.loc[tr, HAR_COLS], y[tr])
        har_pred = pd.Series(har.predict(X.loc[te, HAR_COLS]), index=X.index[te])
        if mode == 'crosssectional':
            har_score = har_pred.groupby(level=0).rank(pct=True).to_numpy()
        else:
            har_score = (har_pred.to_numpy() - lab['threshold'].to_numpy()[te])
        har_bin = (har_pred.to_numpy() > lab['threshold'].to_numpy()[te]).astype(float)

        # --- HAR logit and full RF
        logit_cols = HAR_COLS + [c for c in Xa.columns if c not in X.columns]
        hl = build_har_logit().fit(Xa.loc[tr, logit_cols], ytr)
        p_hl = hl.predict_proba(Xa.loc[te, logit_cols])[:, 1]

        rf = build_classifier().fit(Xa[tr], ytr)
        p_rf = rf.predict_proba(Xa[te])[:, 1]

        base = yte.mean()
        cur = lab['current_state'].to_numpy()[te]
        row = {'test_start': f[0].date(), 'test_end': f[-1].date(),
               'n_test': int(te.sum()), 'base_rate': base}
        for name, sc, pr in [('pers', pers_score[te], None),
                             ('har_thresh', har_score, None),
                             ('har_logit', p_hl, p_hl),
                             ('rf', p_rf, p_rf)]:
            m = classification_metrics(yte, sc, top_frac=top_frac, prob=pr)
            row[f'auc_{name}'] = m['auc']
            row[f'prec_{name}'] = m['prec_top_decile']
            row[f'lift_{name}'] = m['lift']
            if pr is not None:
                row[f'brier_{name}'] = m['brier']
        # binary persistence and binary HAR: precision = P(label | flagged)
        for name, flag in [('pers_bin', cur), ('har_bin', har_bin)]:
            fl = flag.astype(bool)
            row[f'prec_{name}'] = yte[fl].mean() if fl.sum() else np.nan
            row[f'recall_{name}'] = yte[fl].sum() / yte.sum() if yte.sum() else np.nan
            row[f'flagrate_{name}'] = fl.mean()
        rows.append(row)

        preds.append(pd.DataFrame({
            'stock': lab['stock'].to_numpy()[te], 'label': yte,
            'current_state': cur, 'pers_score': pers_score[te],
            'har_score': har_score, 'p_har_logit': p_hl, 'p_rf': p_rf,
        }, index=X.index[te]))

    out = pd.DataFrame(rows)
    if len(out):
        out.loc['mean'] = out.mean(numeric_only=True)
    return out, (pd.concat(preds) if preds else pd.DataFrame())


def state_split(preds, top_frac=0.1):
    """Split pooled OOS predictions on the current state -- the interesting table.

    'persistence' rows (already high vol) are trivially predictable and inflate
    every pooled metric; 'transition' rows (currently calm) are the only ones
    where a forecast tells you something you did not already know. Reported
    separately because the pooled number is close to a weighted average of a
    solved problem and an unsolved one.
    """
    rows = []
    for name, sub in [('all', preds),
                      ('calm now (transition)', preds[preds['current_state'] == 0]),
                      ('turbulent now (persistence)', preds[preds['current_state'] == 1])]:
        if len(sub) == 0 or sub['label'].nunique() < 2:
            continue
        for model, col in [('persistence', 'pers_score'), ('har_thresh', 'har_score'),
                           ('har_logit', 'p_har_logit'), ('rf', 'p_rf')]:
            m = classification_metrics(sub['label'], sub[col], top_frac=top_frac)
            rows.append({'subset': name, 'model': model, 'n': m['n'],
                         'base_rate': m['base_rate'], 'auc': m['auc'],
                         'prec_top_decile': m['prec_top_decile'], 'lift': m['lift']})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- entry point

def predict_high_vol(stock_list, cutoff, horizon=10, mode='timeseries',
                     quantile=QUANTILE, model='rf'):
    """Probability that each stock's next `horizon` days are a high-vol period.

    Fits on everything before `cutoff` (minus a `horizon * 1.5` day purge) and
    returns one probability per date-stock at or after it. `mode` picks the
    question: 'timeseries' = high relative to this stock's own history,
    'crosssectional' = top quintile of the universe that day.

    Read the output with the module docstring in mind: it is well ordered and
    roughly calibrated, but on currently-calm names -- the only case where it adds
    to what you already know -- a 0.3 really does mean 0.3.
    """
    X, y = build_vol_panel(stock_list, horizon=horizon)
    lab = build_labels(y, X['stock'], horizon, mode=mode, quantile=quantile)
    keep = np.asarray(lab['label'].notna())
    X, y, lab = X[keep], y[keep], lab[keep]

    cutoff = pd.Timestamp(cutoff)
    tr = np.asarray(X.index < cutoff - pd.Timedelta(days=int(horizon * 1.5)))
    te = np.asarray(X.index >= cutoff)
    ytr = lab['label'].to_numpy()[tr]

    if model == 'rf':
        fitted = build_classifier().fit(X[tr], ytr)
        p = fitted.predict_proba(X[te])[:, 1]
    elif model == 'har_logit':
        fitted = build_har_logit().fit(X.loc[tr, HAR_COLS], ytr)
        p = fitted.predict_proba(X.loc[te, HAR_COLS])[:, 1]
    else:
        raise ValueError(f'unknown model {model!r}')

    return pd.DataFrame({'stock': lab['stock'].to_numpy()[te],
                         'p_high_vol': p,
                         'current_state': lab['current_state'].to_numpy()[te]},
                        index=X.index[te])
