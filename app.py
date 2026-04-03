import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.linear_model import Ridge
import time
from datetime import datetime

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Minute-by-Minute Predictor", layout="wide")

st.title("📈 Real-Time Stock Market Predictor")
st.sidebar.header("Control Panel")

# Sidebar settings
ticker_list = ["QQQ", "SPY", "NVDA", "TSLA", "AAPL", "BTC-USD", "SLV"]
selected_ticker = st.sidebar.selectbox("Select Ticker", ticker_list)
prediction_window = st.sidebar.slider("Minutes into the Future", 1, 15, 5)
refresh_rate = st.sidebar.slider("Auto-Refresh (Seconds)", 30, 300, 60)

# --- 2. PREDICTION ENGINE ---
def build_model(symbol, window):
    # Fetch 1-minute data for the last 5 days
    # prepost=True is vital for seeing action outside 9:30-4:00 EST
    df = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
    
    if df.empty:
        return None, None, None

    # Adding Technical Indicators (The "Math" your model uses)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    # The Target: What the price will be 'X' minutes from now
    df['Target'] = df['Close'].shift(-window)
    
    # Cleaning data for the model
    features = ['Close', 'RSI', 'VWAP', 'EMA_9', 'Volatility']
    data = df.dropna()
    
    if data.empty:
        return None, None, df

    X = data[features]
    y = data['Target']
    
    # Training the Ridge Regression model (Good for noisy stock data)
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    
    # Get the most recent data point to predict the next 'window' minutes
    latest_row = df[features].tail(1)
    current_price = df['Close'].iloc[-1]
    prediction = model.predict(latest_row)[0]
    
    return current_price, prediction, df

# --- 3. THE LIVE DASHBOARD ---
# This "placeholder" allows the app to overwrite itself instead of scrolling down
placeholder = st.empty()

while True:
    current_p, predicted_p, full_df = build_model(selected_ticker, prediction_window)
    
    if current_p is not None:
        with placeholder.container():
            # Calculate the expected move
            diff = predicted_p - current_p
            pct_change = (diff / current_p) * 100
            
            # Layout: Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_p:.2f}")
            col2.metric(f"Target (+{prediction_window}m)", f"${predicted_p:.2f}", f"{diff:.2f} ({pct_change:.2f}%)")
            
            # Logic for the "Star Rating"
            rating = 5 
            if pct_change > 0.04: rating = 9   # Strong Buy
            elif pct_change > 0.01: rating = 7 # Buy
            elif pct_change < -0.04: rating = 1 # Strong Sell
            elif pct_change < -0.01: rating = 3 # Sell
            
            col3.write(f"### Confidence Score")
            col3.write("⭐" * int(rating))

            # Visual: Recent Price Chart
            st.subheader(f"Recent {selected_ticker} Trend (1m Intervals)")
            st.line_chart(full_df['Close'].tail(60))
            
            st.caption(f"Last Engine Refresh: {datetime.now().strftime('%H:%M:%S')}")
            
    else:
        st.warning("Waiting for market data... check your ticker or internet connection.")

    # Pause the script before the next update
    time.sleep(refresh_rate)