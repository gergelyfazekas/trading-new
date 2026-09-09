"""Grid-search driver for validating technical-level parameters.

Two nested loops, matching the cost structure of tech_levels.py's mechanics:
  - outer: parameters that change scipy.signal.find_peaks itself (prominence,
    distance, height, rel_height, consider_volume, volume_height, volume_prominence)
    -- these force touches to be re-detected.
  - inner: tech_width -- only re-clusters touches already found, cheap.

Parallelizes across stocks (not across combos) with ProcessPoolExecutor, following
the pattern in genetic_algo.py's run_single_portfolio/simulate. A stock's full
close/volume history is the natural unit of work, and every combo in the grid
shares the outer loop's find_touches result within one process. Writes one CSV
per stock into config.tech_level_folder, matching the per-ticker convention in
StockList.save_data/load_data (stock_class.py).
"""

import itertools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from tech_levels import find_touches, build_levels_causal, score_touches, levels_to_rows
from config import tech_level_folder


DEFAULT_GRID = {
    'outer': {
        # calibrated against real peak-prominence distributions on log(close)
        # (find_touches works in log space, so prominence is a fractional move).
        # Universe p50 ~0.010, p75 ~0.022, p90 ~0.047; this range spans p50-p90,
        # which works out to roughly 27 touches/year at 0.01 down to 11/year at
        # 0.05 with distance=10. Log space also compresses the spread between
        # calm and volatile names (p50 runs 0.0066 for KO to 0.0167 for AMAT,
        # ~2.5x, vs the ~3x seen under the old absolute scaling), so one grid
        # covers the universe more evenly than it used to.
        'prominence': [0.01, 0.02, 0.03, 0.05],
        'distance': [5, 10, 20],
        'consider_volume': [False],
        # NB: rel_height only affects find_peaks when a `width` constraint is
        # also passed, which we never do -- it was previously dead weight in
        # the grid and has been dropped rather than left in as a no-op axis.
    },
    'tech_width': [0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05],
}


def expand_outer(outer_spec: dict) -> list:
    """Cartesian product of the outer (find_peaks-affecting) param lists."""
    keys = list(outer_spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(outer_spec[k] for k in keys))]


def expand_outer_with_flag(base_spec: dict, flag_name: str, flag_defaults: dict, gated_spec: dict) -> list:
    """Like expand_outer, but for a param (e.g. consider_volume) that gates a
    set of other params which are inert when it's off (volume_height,
    volume_prominence do nothing when consider_volume=False). Blindly crossing
    those into the grid would just produce duplicate, differently-labeled combos
    for every False case. Returns one {flag_name: False, **flag_defaults} combo
    per base combo, plus base combos crossed with gated_spec at {flag_name: True}.
    """
    base_combos = expand_outer(base_spec)
    gated_combos = expand_outer(gated_spec)
    combos = []
    for base in base_combos:
        combos.append({**base, flag_name: False, **flag_defaults})
        for gated in gated_combos:
            combos.append({**base, flag_name: True, **gated})
    return combos


def run_stock_grid(ticker, close: pd.Series, outer_combos: list, tech_widths: list,
                    volume: pd.Series = None) -> pd.DataFrame:
    """Run the full grid for one stock: find_touches once per outer combo, then
    sweep tech_width cheaply against those same touches."""
    rows = []
    for outer in outer_combos:
        consider_volume = outer.get('consider_volume', False)
        volume_height = outer.get('volume_height', 0.0)
        volume_prominence = outer.get('volume_prominence', 0.0)
        peak_kwargs = {k: v for k, v in outer.items()
                       if k not in ('consider_volume', 'volume_height', 'volume_prominence')}

        touches = find_touches(close, volume=volume, consider_volume=consider_volume,
                                volume_height=volume_height, volume_prominence=volume_prominence,
                                **peak_kwargs)

        for tech_width in tech_widths:
            levels = build_levels_causal(touches, tech_width)
            for level in levels:
                score_touches(level, close, volume=volume)
            full_combo = {
                'tech_width': tech_width, 'consider_volume': consider_volume,
                'volume_height': volume_height, 'volume_prominence': volume_prominence,
                **peak_kwargs,
            }
            rows.extend(levels_to_rows(ticker, full_combo, levels))

    return pd.DataFrame(rows)


def _worker(args):
    ticker, close, volume, outer_combos, tech_widths, output_folder = args
    log = run_stock_grid(ticker, close, outer_combos, tech_widths, volume=volume)
    os.makedirs(output_folder, exist_ok=True)
    log.to_csv(os.path.join(output_folder, f'{ticker}.csv'), index=False)
    return ticker, len(log)


def run_search(stock_list, grid: dict = None, output_folder: str = tech_level_folder, max_workers: int = 8):
    """Run the grid search across every stock in stock_list, writing one CSV per
    stock into output_folder.

    stock_list: a stock_class.StockList already loaded with data, e.g.
        s = StockList(ticker_list); s.load_data(path=gold)
    grid: {'outer': {param: [values...], ...}, 'tech_width': [values...]};
        defaults to DEFAULT_GRID. grid['outer'] may also be a pre-built list of
        combo dicts (e.g. from expand_outer_with_flag) instead of a spec, for
        grids that need to avoid a blind Cartesian product. Pass a smaller
        grid / ticker subset for a quick sanity run before the full search.
    """
    grid = grid or DEFAULT_GRID
    outer = grid['outer']
    outer_combos = outer if isinstance(outer, list) else expand_outer(outer)
    tech_widths = grid['tech_width']

    jobs = []
    for ticker in stock_list.tickers:
        data = stock_list[ticker].data
        close = data['close']
        volume = data['volume'] if 'volume' in data.columns else None
        jobs.append((ticker, close, volume, outer_combos, tech_widths, output_folder))

    if mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('spawn', force=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            ticker, n_rows = future.result()
            print(f'{ticker}: {n_rows} log rows')
            results.append((ticker, n_rows))

    return results


if __name__ == '__main__':
    from stock_class import StockList
    from config import ticker_list, gold

    s = StockList(ticker_list)
    s.load_data(path=gold)
    run_search(s)
