"""Streamlit inspector for technical-level param combos.

Pick a stock and up to two param combos, see the price chart with each combo's
levels drawn from their birth date, hits marked green and misses marked red, plus
the four local/global x in-sample/out-of-sample scores side by side -- so a good-
and a bad-scoring combo can be compared directly instead of just trusting a number.

Run with: streamlit run tech_level_app.py
To point it at a different gold/tech-level dataset (e.g. an ad hoc ticker
subset instead of config.py's main universe), set env vars before launching:
    TECH_LEVEL_GOLD=data/gold TECH_LEVEL_LOG=data/tech_levels TECH_LEVEL_TICKERS=AAPL,MSFT,NVDA,AMZN,GOOGL \\
        streamlit run tech_level_app.py
"""

import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import ticker_list as default_ticker_list, gold as default_gold, tech_level_folder as default_tech_level_folder
from stock_class import StockList
from tech_level_scoring import load_log, score

GOLD = os.environ.get('TECH_LEVEL_GOLD', default_gold)
TECH_LEVEL_FOLDER = os.environ.get('TECH_LEVEL_LOG', default_tech_level_folder)
_tickers_env = os.environ.get('TECH_LEVEL_TICKERS')
TICKER_LIST = [t.strip() for t in _tickers_env.split(',')] if _tickers_env else default_ticker_list


st.set_page_config(layout="wide", page_title="Tech level inspector")
st.title("Technical level inspector")


@st.cache_resource
def get_stock_list():
    s = StockList(TICKER_LIST)
    s.load_data(path=GOLD)
    return s


@st.cache_data
def get_log():
    return load_log(TECH_LEVEL_FOLDER)


try:
    log = get_log()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun tech_level_search.run_search(...) first to generate the touch log.")
    st.stop()

stock_list = get_stock_list()

with st.sidebar:
    ticker = st.selectbox("Stock", sorted(log['stock'].unique()))

    all_dates = pd.concat([log['birth_date'], log['touch_date']]).dropna()
    min_date, max_date = all_dates.min().date(), all_dates.max().date()
    default_cutoff = min_date + (max_date - min_date) * 3 // 4
    cutoff = st.date_input("Cutoff date (in-sample / out-of-sample split)",
                            value=default_cutoff, min_value=min_date, max_value=max_date)

    min_touches = st.number_input("Minimum touches per level (0 = off)", min_value=0, value=0, step=1)

scores = score(log, cutoff_date=pd.Timestamp(cutoff), min_touches=min_touches)
if ticker in scores.index.get_level_values('stock'):
    stock_scores = scores.xs(ticker, level='stock').sort_values('local_oos', ascending=False)
else:
    stock_scores = scores.iloc[0:0]

if stock_scores.empty:
    st.warning(f"No combos have any levels for {ticker} at this cutoff/filter.")
    st.stop()

combo_options = stock_scores.index.tolist()

pick_col1, pick_col2 = st.columns(2)
with pick_col1:
    combo_a = st.selectbox("Combo A (defaults to best local OOS)", combo_options, index=0, key='combo_a')
with pick_col2:
    combo_b = st.selectbox("Combo B (defaults to worst local OOS)", combo_options,
                            index=len(combo_options) - 1, key='combo_b')


def render_combo(container, ticker, combo_id, log, close, scores_row):
    with container:
        st.subheader(combo_id, divider=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, key, label in [
            (m1, 'local_is', 'local IS'), (m2, 'local_oos', 'local OOS'),
            (m3, 'global_loo_is', 'global IS'), (m4, 'global_loo_oos', 'global OOS'),
        ]:
            val = scores_row[key]
            col.metric(label, f"{val:.0%}" if pd.notna(val) else "n/a")

        combo_log = log[(log['stock'] == ticker) & (log['combo_id'] == combo_id)]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(close.index, close.values, color='black', linewidth=0.8, zorder=1)

        for level_id, level_rows in combo_log.groupby('level_id'):
            band_low = level_rows['band_low'].iloc[0]
            band_high = level_rows['band_high'].iloc[0]
            birth = level_rows['birth_date'].iloc[0].date()
            ax.axhspan(band_low, band_high, xmin=0, xmax=1, color='tab:blue', alpha=0.08, zorder=0)
            ax.hlines([band_low, band_high], xmin=birth, xmax=close.index.max(),
                      color='tab:blue', alpha=0.5, linewidth=1, zorder=2)

            for _, row in level_rows.dropna(subset=['touch_date']).iterrows():
                touch_date = row['touch_date'].date()
                y = (band_low + band_high) / 2
                if row['outcome'] == 'hit':
                    ax.scatter([touch_date], [y], color='green', marker='^', s=40, zorder=5)
                elif row['outcome'] == 'miss':
                    ax.scatter([touch_date], [y], color='red', marker='x', s=40, zorder=5)

        ax.axvline(cutoff, color='gray', linestyle='--', linewidth=1, label='cutoff')
        ax.set_title(f"{ticker} -- {combo_id}")
        ax.legend(loc='upper left', fontsize='small')
        st.pyplot(fig)
        plt.close(fig)


close = stock_list[ticker].data['close']
chart_col1, chart_col2 = st.columns(2)
render_combo(chart_col1, ticker, combo_a, log, close, stock_scores.loc[combo_a])
render_combo(chart_col2, ticker, combo_b, log, close, stock_scores.loc[combo_b])
