import streamlit as st
st.set_page_config(page_title="Pivot Analyzer", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# ----------------- App Setup ------------------ #
# Sidebar configuration
st.sidebar.title("📁 Upload & Settings")

# Theme selector
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])

# Apply dark theme CSS
if theme == "Dark":
    st.markdown("""
        <style>
            .main {
                background-color: #0e1117;
                color: white;
            }
            .css-18e3th9, .css-1d391kg, .css-hxt7ib, .css-10trblm {
                background-color: #0e1117 !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Pivot window selector
window = st.sidebar.selectbox("Pivot Detection Window", [2, 3, 5, 8, 10], index=4)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ----------------- Main UI ------------------ #
st.markdown("### 📊 Pivot Analyzer Visualization")

if uploaded_file:
    filename = uploaded_file.name.replace("-EQ.csv", "").upper()
    
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df = df.rename(columns={'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.reset_index(drop=True, inplace=True)
        return df

    df = load_data(uploaded_file)

    def is_pivot(candle, window, breakout_indices, realtime=False):
        if candle not in breakout_indices or candle - window < 0:
            return 0
        pivotHigh = 1
        pivotLow = 2
        right_window = 0 if (realtime and candle + window >= len(df)) else window
        for i in range(candle - window, candle + right_window + 1):
            if i == candle:
                continue
            if df.iloc[candle].low > df.iloc[i].low:
                pivotLow = 0
            if df.iloc[candle].high < df.iloc[i].high:
                pivotHigh = 0
        if pivotHigh and pivotLow:
            return 3
        elif pivotHigh:
            return pivotHigh
        elif pivotLow:
            return pivotLow
        else:
            return 0

    # Detect breakout points
    breakout_indices = [0]
    last_high = df.loc[0, 'high']
    last_low = df.loc[0, 'low']
    for i in range(1, len(df)):
        high = df.loc[i, 'high']
        low = df.loc[i, 'low']
        if high > last_high or low < last_low:
            breakout_indices.append(i)
            last_high = high
            last_low = low

    # Detect pivots
    df['isPivot'] = df.apply(
        lambda x: is_pivot(
            x.name, window, breakout_indices,
            realtime=(x.name >= len(df) - window)
        ),
        axis=1
    )

    df['high_pivot'] = df['isPivot'].isin([1, 3])
    df['low_pivot'] = df['isPivot'].isin([2, 3])
    events = []
    for idx in df.index:
        if df.loc[idx, 'high_pivot']:
            events.append({'index': idx, 'type': 'high', 'value': df.loc[idx, 'high']})
        if df.loc[idx, 'low_pivot']:
            events.append({'index': idx, 'type': 'low', 'value': df.loc[idx, 'low']})
    events.sort(key=lambda x: x['index'])

    valid_pivots = []
    last_index = -float('inf')
    last_type = None
    for event in events:
        current_idx = event['index']
        current_type = event['type']
        current_val = event['value']
        if not valid_pivots:
            valid_pivots.append(event)
            last_index = current_idx
            last_type = current_type
        else:
            if current_type == last_type:
                if (current_type == 'high' and current_val > valid_pivots[-1]['value']) or \
                   (current_type == 'low' and current_val < valid_pivots[-1]['value']):
                    valid_pivots.pop()
                    valid_pivots.append(event)
                    last_index = current_idx
            else:
                if current_idx - last_index >= 3:
                    valid_pivots.append(event)
                    last_index = current_idx
                    last_type = current_type

    df['isPivot'] = 0
    for pivot in valid_pivots:
        idx = pivot['index']
        df.at[idx, 'isPivot'] = 1 if pivot['type'] == 'high' else 2

    df['pivot_label'] = None
    last_high = last_low = None
    for idx in df.index:
        if df.loc[idx, 'isPivot'] == 1:
            label = 'HH' if last_high is not None and df.loc[idx, 'high'] > last_high else 'LH'
            df.at[idx, 'pivot_label'] = label
            last_high = df.loc[idx, 'high']
        elif df.loc[idx, 'isPivot'] == 2:
            label = 'HL' if last_low is not None and df.loc[idx, 'low'] > last_low else 'LL'
            df.at[idx, 'pivot_label'] = label
            last_low = df.loc[idx, 'low']

    df['p_high'] = df.apply(lambda x: x['high'] if x['isPivot'] == 1 else np.nan, axis=1)
    df['p_low'] = df.apply(lambda x: x['low'] if x['isPivot'] == 2 else np.nan, axis=1)

    dfpl = df.tail(200).copy()

    def plot_chart():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                       gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})

        # Candlesticks
        for idx, row in dfpl.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            ax1.plot([idx, idx], [row['low'], row['high']], color='black')
            ax1.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=3.0)

        # Pivot labels
        colors = {'HH': 'lime', 'LH': 'yellow', 'LL': 'black', 'HL': 'blue'}
        for label, color in colors.items():
            subset = dfpl[dfpl['pivot_label'] == label]
            y_vals = subset['high'] if label in ['HH', 'LH'] else subset['low']
            ax1.scatter(subset.index, y_vals, color=color, label=label, zorder=5)

        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.yaxis.set_ticks_position('right')
        ax1.yaxis.set_label_position('right')

        # Watermark
        ax1.text(0.5, 0.5, filename, fontsize=48, color='gray',
                 ha='center', va='center', alpha=0.3, transform=ax1.transAxes)

        # Grid and date ticks every 25 rows
        ticks = dfpl.index[::25]
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(dfpl['date'].dt.strftime('%Y-%m-%d').iloc[::25], rotation=45, fontsize=8)
        ax1.grid(True, which='major', axis='x')

        # Volume bars
        volume_colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in dfpl.iterrows()]
        ax2.bar(dfpl.index, dfpl['volume'], color=volume_colors, width=0.7)
        ax2.set_ylabel("Volume")
        ax2.yaxis.set_ticks_position('right')
        ax2.yaxis.set_label_position('right')

        plt.xlabel("Date")
        ax1.set_xlim(left=dfpl.index.min() - 0.5, right=dfpl.index.max() + 3)
        plt.tight_layout()
        return fig

    fig = plot_chart()
    st.pyplot(fig)

    # Option to download chart
    chart_format = st.sidebar.selectbox("Download Chart As", ["None", "PNG", "JPEG"])
    if chart_format != "None":
        buf = io.BytesIO()
        fig.savefig(buf, format=chart_format.lower())
        st.sidebar.download_button(
            label=f"📥 Download Chart ({chart_format})",
            data=buf.getvalue(),
            file_name=f"{filename}_pivot_chart.{chart_format.lower()}",
            mime=f"image/{chart_format.lower()}"
        )
else:
    st.info("👈 Please upload a CSV file to begin.")
