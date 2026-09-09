"""Turn a tech-level touch log (from tech_level_search.py) into scores.

Every level and touch is stored unfiltered by the search step; everything here
is a query over that log, so changing the cutoff date, the min-touches filter,
or the grouping doesn't require re-running any peak detection.

Four scores per (stock, combo), all derived from the same log:
  - local_is / local_oos: this stock's own hit rate before/after the cutoff date
  - global_loo_is / global_loo_oos: hit rate pooled across every *other* stock
    (leave-one-out), so a stock's own result can't inflate its own corroboration

A level only counts toward OOS scoring if it was already born on/before the
cutoff -- a level born after the cutoff isn't a fair "did the known signal hold"
test either way, so it's excluded from both IS and OOS scoring.
"""

import glob
import os

import pandas as pd


COMBO_COLS_GUESS = ['combo_id']  # grouping key; param columns ride along for free via combo_id


def load_log(tech_level_folder: str) -> pd.DataFrame:
    """Concatenate every per-stock CSV written by tech_level_search.run_search."""
    frames = []
    for path in sorted(glob.glob(os.path.join(tech_level_folder, '*.csv'))):
        df = pd.read_csv(path, parse_dates=['birth_date', 'touch_date'])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'No log CSVs found in {tech_level_folder}')
    return pd.concat(frames, ignore_index=True)


def _level_touch_counts(log: pd.DataFrame) -> pd.DataFrame:
    """One row per level: stock, combo_id, birth_date, hits, misses (touches
    with outcome=None, i.e. zero post-birth touches, count as hits=0, misses=0)."""
    has_outcome = log['outcome'].isin(['hit', 'miss'])
    counted = log[has_outcome]
    per_level = (
        counted.groupby(['stock', 'combo_id', 'level_id', 'birth_date'])['outcome']
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=['hit', 'miss'], fill_value=0)
        .reset_index()
    )
    # bring back levels that had zero post-birth touches at all
    all_levels = log[['stock', 'combo_id', 'level_id', 'birth_date']].drop_duplicates()
    per_level = all_levels.merge(per_level, on=['stock', 'combo_id', 'level_id', 'birth_date'], how='left')
    per_level[['hit', 'miss']] = per_level[['hit', 'miss']].fillna(0)
    return per_level


def _touches_after_cutoff(log: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Per-level hit/miss counts using only touches after cutoff, restricted to
    levels already born on/before cutoff (the OOS-eligibility rule)."""
    eligible = log[log['birth_date'] <= cutoff]
    after = eligible[eligible['outcome'].isin(['hit', 'miss']) & (eligible['touch_date'] > cutoff)]
    per_level = (
        after.groupby(['stock', 'combo_id', 'level_id'])['outcome']
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=['hit', 'miss'], fill_value=0)
        .reset_index()
    )
    all_eligible_levels = eligible[['stock', 'combo_id', 'level_id']].drop_duplicates()
    per_level = all_eligible_levels.merge(per_level, on=['stock', 'combo_id', 'level_id'], how='left')
    per_level[['hit', 'miss']] = per_level[['hit', 'miss']].fillna(0)
    return per_level


def _touches_up_to_cutoff(log: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Per-level hit/miss counts using only levels+touches at or before cutoff
    (used for in-sample scoring / combo selection)."""
    eligible = log[log['birth_date'] <= cutoff]
    before = eligible[eligible['outcome'].isin(['hit', 'miss']) & (eligible['touch_date'] <= cutoff)]
    per_level = (
        before.groupby(['stock', 'combo_id', 'level_id'])['outcome']
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=['hit', 'miss'], fill_value=0)
        .reset_index()
    )
    all_eligible_levels = eligible[['stock', 'combo_id', 'level_id']].drop_duplicates()
    per_level = all_eligible_levels.merge(per_level, on=['stock', 'combo_id', 'level_id'], how='left')
    per_level[['hit', 'miss']] = per_level[['hit', 'miss']].fillna(0)
    return per_level


def _apply_min_touches(per_level: pd.DataFrame, min_touches: int) -> pd.DataFrame:
    if min_touches <= 0:
        return per_level
    return per_level[(per_level['hit'] + per_level['miss']) >= min_touches]


def _hit_rate(per_level: pd.DataFrame, group_cols: list) -> pd.Series:
    grouped = per_level.groupby(group_cols)[['hit', 'miss']].sum()
    total = grouped['hit'] + grouped['miss']
    return (grouped['hit'] / total).where(total > 0)


def score(log: pd.DataFrame, cutoff_date, min_touches: int = 0) -> pd.DataFrame:
    """Return a DataFrame indexed by (stock, combo_id) with columns:
    local_is, local_oos, global_loo_is, global_loo_oos,
    n_touches_is, n_touches_oos (the actual denominators behind local_is/local_oos --
    filter on these, not the whole-history n_touches, or a 3-for-3 OOS fluke will
    look identical to a real result), plus n_levels/n_touches as whole-history
    diagnostics.

    min_touches: minimum post-birth touches (hits+misses) a level needs to count
    toward any aggregate score. Disabled (0) by default -- everything is counted
    first; this is a query-time filter, not baked into collection.
    """
    is_levels = _apply_min_touches(_touches_up_to_cutoff(log, cutoff_date), min_touches)
    oos_levels = _apply_min_touches(_touches_after_cutoff(log, cutoff_date), min_touches)

    local_is = _hit_rate(is_levels, ['stock', 'combo_id'])
    local_oos = _hit_rate(oos_levels, ['stock', 'combo_id'])

    n_touches_is = (is_levels.groupby(['stock', 'combo_id'])[['hit', 'miss']].sum().sum(axis=1)
                     .rename('n_touches_is'))
    n_touches_oos = (oos_levels.groupby(['stock', 'combo_id'])[['hit', 'miss']].sum().sum(axis=1)
                      .rename('n_touches_oos'))

    # global (leave-one-out): pooled hit/miss per combo, minus each stock's own contribution
    def loo_hit_rate(per_level: pd.DataFrame) -> pd.Series:
        by_stock_combo = per_level.groupby(['stock', 'combo_id'])[['hit', 'miss']].sum()
        pooled = by_stock_combo.groupby('combo_id')[['hit', 'miss']].transform('sum')
        loo = pooled - by_stock_combo
        total = loo['hit'] + loo['miss']
        return (loo['hit'] / total).where(total > 0)

    global_loo_is = loo_hit_rate(is_levels)
    global_loo_oos = loo_hit_rate(oos_levels)

    all_levels = log[['stock', 'combo_id', 'level_id']].drop_duplicates()
    n_levels = all_levels.groupby(['stock', 'combo_id']).size().rename('n_levels')
    n_touches = (
        log[log['outcome'].isin(['hit', 'miss'])]
        .groupby(['stock', 'combo_id']).size().rename('n_touches')
    )

    result = pd.DataFrame({
        'local_is': local_is, 'local_oos': local_oos,
        'global_loo_is': global_loo_is, 'global_loo_oos': global_loo_oos,
        'n_touches_is': n_touches_is, 'n_touches_oos': n_touches_oos,
    })
    result = result.join(n_levels, how='outer').join(n_touches, how='outer')
    count_cols = ['n_levels', 'n_touches', 'n_touches_is', 'n_touches_oos']
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    return result.sort_index()
