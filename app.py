import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import Ridge
import time
from datetime import datetime

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Real-Time Stock Predictor")

ticker_list = ["QQQ", "SPY", "NVDA", "TSLA", "AAPL", "BTC-USD", "SLV"]
selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)

def build_model(symbol, window):
    df = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
    if df.empty: return None, None, None

    # Manual RSI (to avoid the pandas-ta crash)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Simple Moving Average
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['Target'] = df['Close'].shift(-window)
    
    features = ['Close', 'RSI', 'SMA_9']
    data = df.dropna()
    if data.empty: return None, None, df

    model = Ridge(alpha=1.0)
    model.fit(data[features], data['Target'])
    
    latest_row = df[features].tail(1)
    current_price = df['Close'].iloc[-1]
    prediction = model.predict(latest_row)[0]
    
    return current_price, prediction, df

placeholder = st.empty()
while True:
    current_p, predicted_p, full_df = build_model(selected_ticker, 5)
    if current_p is not None:
        with placeholder.container():
            st.metric("Current Price", f"${current_p:.2f}")
            st.metric("Target (5m)", f"${predicted_p:.2f}", f"{predicted_p - current_p:.2f}")
            st.line_chart(full_df['Close'].tail(50))
    time.sleep(60)
