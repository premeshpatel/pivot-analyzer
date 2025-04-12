# 📊 Pivot Analyzer

A simple Streamlit web app to visualize price pivots (HH, HL, LH, LL) based on custom window sizes.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
streamlit run app.py
```

## 📁 File Format

Make sure your CSV contains these columns:

- `date` (optional — index is used if not present)
- `O`, `H`, `L`, `C`, `V` → Open, High, Low, Close, Volume

## ⚙️ Features

- Detects pivot highs/lows (with real-time support)
- Interactive candlestick chart with color-coded pivots
- PNG download of the chart
- Dark/light theme (via `.streamlit/config.toml`)
