import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import Ridge
import time
from datetime import datetime

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Predictor")

# Sidebar for controls
ticker_list = ["QQQ", "SPY", "NVDA", "TSLA", "AAPL", "BTC-USD", "SLV"]
selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
prediction_minutes = st.sidebar.slider("Minutes into Future", 1, 15, 5)

# --- 2. THE ENGINE ---
def build_model(symbol, window):
    # Get 1-minute data
    df = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
    
    if df.empty:
        return None, None, None

    # Manual RSI Calculation (Stable for Cloud)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple Moving Average
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    
    # Target: Price 'X' minutes in the future
    df['Target'] = df['Close'].shift(-window)
    
    features = ['Close', 'RSI', 'SMA_9']
    data = df.dropna()
    
    if data.empty:
        return None, None, df

    # Train a quick Ridge model
    model = Ridge(alpha=1.0)
    model.fit(data[features], data['Target'])
    
    # CRITICAL FIX: Convert to float to avoid TypeError
    latest_row = df[features].tail(1)
    current_price = float(df['Close'].iloc[-1]) 
    prediction = float(model.predict(latest_row)[0])
    
    return current_price, prediction, df

# --- 3. THE DASHBOARD ---
placeholder = st.empty()

while True:
    current_p, predicted_p, full_df = build_model(selected_ticker, prediction_minutes)
    
    if current_p is not None:
        with placeholder.container():
            diff = predicted_p - current_p
            
            # Display Metrics
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"${current_p:.2f}")
            col2.metric(f"Target (+{prediction_minutes}m)", f"${predicted_p:.2f}", f"{diff:.2f}")
            
            # Display Trend Chart
            st.subheader(f"Recent {selected_ticker} Movement")
            st.line_chart(full_df['Close'].tail(60))
            
            st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    else:
        st.warning("Connecting to market data... (If this persists, check the 'Manage App' logs for Rate Limits)")

    # Wait 60 seconds before refreshing
    time.sleep(60)
