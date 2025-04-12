
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(page_title="Pivot Analyzer", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.title("📁 Pivot Analyzer")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    window = st.selectbox("Pivot Detection Window", [2, 3, 5, 8, 10], index=2)

# === Main View ===
st.markdown("### 📊 Pivot Chart Visualization")

if uploaded_file:
    filename = uploaded_file.name.replace("-EQ.csv", "").split(".")[0].upper()
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns={'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
    df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
    df.reset_index(drop=True, inplace=True)

    # === Pivot Detection Logic with Real-Time Handling ===
    def is_pivot(idx, w, realtime=False):
        if idx - w < 0:
            return 0

        right_window = 0 if (realtime and idx + w >= len(df)) else w
        pivot_high = True
        pivot_low = True

        for i in range(idx - w, idx + right_window + 1):
            if i == idx or i >= len(df):
                continue
            if df.loc[idx, 'low'] > df.loc[i, 'low']:
                pivot_low = False
            if df.loc[idx, 'high'] < df.loc[i, 'high']:
                pivot_high = False

        if pivot_high and pivot_low:
            return 3
        elif pivot_high:
            return 1
        elif pivot_low:
            return 2
        return 0

    df['isPivot'] = [is_pivot(i, window) for i in df.index]
    df['pivot_label'] = None
    last_high = last_low = None

    # === Label Pivots ===
    df['pivot_label'] = None
    last_high = last_low = None
    for i in df.index:
        if df.loc[i, 'isPivot'] == 1:
            label = 'HH' if last_high is None or df.loc[i, 'high'] > last_high else 'LH'
            last_high = df.loc[i, 'high']
            df.at[i, 'pivot_label'] = label
        elif df.loc[i, 'isPivot'] == 2:
            label = 'HL' if last_low is None or df.loc[i, 'low'] > last_low else 'LL'
            last_low = df.loc[i, 'low']
            df.at[i, 'pivot_label'] = label

    df['p_high'] = df.apply(lambda x: x['high'] if x['isPivot'] == 1 else np.nan, axis=1)
    df['p_low'] = df.apply(lambda x: x['low'] if x['isPivot'] == 2 else np.nan, axis=1)

    # === Plotting ===
    dfpl = df.tail(200).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, 
                                   gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})

    for idx, row in dfpl.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax1.plot([idx, idx], [row['low'], row['high']], color='black')
        ax1.plot([idx, idx], [row['open'], row['close']], color=color, linewidth=3.0)

    colors = {'HH': 'lime', 'LH': 'yellow', 'LL': 'black', 'HL': 'blue'}
    for label, color in colors.items():
        subset = dfpl[dfpl['pivot_label'] == label]
        y_vals = subset['high'] if label in ['HH', 'LH'] else subset['low']
        ax1.scatter(subset.index, y_vals, color=color, label=label, zorder=5)

    ax1.set_ylabel("Price")
    ax1.grid(True, which='major', axis='both')
    ax1.legend()
    ax1.yaxis.set_ticks_position('right')
    ax1.yaxis.set_label_position('right')

    # X-axis date formatting and grid lines every 25
    tick_interval = 25
    xticks = dfpl.index[::tick_interval]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(dfpl['date'].dt.strftime('%Y-%m-%d').iloc[::tick_interval], rotation=45)

    # === Volume Chart with Grid ===
    volume_colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in dfpl.iterrows()]
    ax2.bar(dfpl.index, dfpl['volume'], color=volume_colors, width=0.7)
    ax2.set_ylabel("Volume")
    ax2.grid(True)
    ax2.yaxis.set_ticks_position('right')
    ax2.yaxis.set_label_position('right')

    # Watermark
    ax1.text(0.5, 0.5, filename, transform=ax1.transAxes,
             fontsize=80, color='grey', alpha=0.3,
             ha='center', va='center', rotation=0)

    ax1.set_xlim(left=dfpl.index.min() - 0.5, right=dfpl.index.max() + 3)
    ax2.set_xlabel("Index")
    plt.tight_layout()

    # === Display Chart ===
    st.pyplot(fig)

    # === PNG Download Button ===
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="📥 Download Chart as PNG",
        data=buf.getvalue(),
        file_name=f"{filename}_pivot_chart.png",
        mime="image/png"
    )
