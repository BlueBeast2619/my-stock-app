import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
import time
from datetime import datetime

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Pro Stock Predictor", layout="wide")
st.title("📈 AI Stock Predictor (Fast-Load Edition)")

ticker_list = ["QQQ", "SPY", "NVDA", "TSLA", "AAPL", "BTC-USD", "SLV"]
selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
prediction_minutes = st.sidebar.slider("Minutes into Future", 1, 15, 5)

# --- 2. THE CACHE (Speed Improvement) ---
@st.cache_data(ttl=3600)  # Downloads historical data only once per hour
def get_historical_data(symbol):
    return yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)

# --- 3. THE ENGINE ---
def build_model(symbol, window):
    # Fetch data using the cache
    df_raw = get_historical_data(symbol)
    
    if df_raw.empty:
        return None, None

    # Fix MultiIndex column issue immediately
    df = df_raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Technical Indicators
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['Volatility'] = df['Close'].rolling(window=15).std()
    df['Momentum'] = df['Close'].diff(10)
    
    # Volume Pressure
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Pressure'] = df['Volume'] / df['Vol_Avg']
    
    # Target
    df['Target'] = df['Close'].shift(-window)
    
    # Features
    features = ['Close', 'RSI', 'SMA_9', 'Volatility', 'Momentum', 'Vol_Pressure']
    data = df.dropna()
    
    if data.empty:
        return None, df

    model = Ridge(alpha=1.0)
    model.fit(data[features], data['Target'])
    
    # Extract Final Metrics
    latest_row = data[features].tail(1)
    
    metrics = {
        "price": float(df['Close'].values.flatten()[-1]),
        "vol_p": float(df['Vol_Pressure'].values.flatten()[-1]),
        "momentum": float(df['Momentum'].values.flatten()[-1]),
        "prediction": float(model.predict(latest_row).flatten()[0])
    }
    
    return metrics, df

# --- 4. THE DASHBOARD ---
placeholder = st.empty()

while True:
    m, full_df = build_model(selected_ticker, prediction_minutes)
    
    if m is not None:
        with placeholder.container():
            diff = m['prediction'] - m['price']
            
            # Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${m['price']:.2f}")
            
            m_color = "normal" if m['momentum'] > 0 else "inverse"
            c2.metric("10m Momentum", f"{m['momentum']:.2f}", delta_color=m_color)
            
            c3.metric("Vol Pressure", f"{m['vol_p']:.2f}x")
            c4.metric(f"Target (+{prediction_minutes}m)", f"${m['prediction']:.2f}", f"{diff:.2f}")
            
            # Chart Row
            st.subheader(f"Live {selected_ticker} Analysis")
            st.line_chart(full_df['Close'].tail(60))
            
            st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} | Cache active for 1 hour.")
    
    else:
        st.warning("Fetching Market Data... (Check logs if this stays for >2 mins)")

    time.sleep(60)
