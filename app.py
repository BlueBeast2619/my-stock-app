import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import time

# --- PWA CONFIGURATION ---
# This must be the very first Streamlit command
st.set_page_config(page_title="Vigilant Pulse AI", page_icon="📈", layout="wide")

# Inject Manifest for Android/Bubblewrap
# We use a try/except so if the file is missing, the app doesn't hang
try:
    st.markdown(
        f'<link rel="manifest" href="/app/static/manifest.json">',
        unsafe_allow_html=True
    )
except Exception:
    pass

# --- CORE FUNCTIONS ---

@st.cache_data(ttl=60)
def fetch_data(ticker):
    data = yf.download(ticker, period="1d", interval="1m", prepost=True)
    if data.empty:
        return None
    # Fix for MultiIndex columns in newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_indicators(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Momentum
    df['Momentum'] = df['Close'].diff(10)
    return df.dropna()

def predict_price(df):
    # Prepare features for the next 5 minutes
    df['Target'] = df['Close'].shift(-5)
    train = df.dropna()
    
    X = train[['Close', 'RSI', 'VWAP', 'Momentum']]
    y = train['Target']
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    current_features = df[['Close', 'RSI', 'VWAP', 'Momentum']].iloc[[-1]]
    prediction = model.predict(current_features)[0]
    return prediction

# --- USER INTERFACE ---

st.title("📈 Vigilant Pulse AI")
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Enter Ticker (e.g. QQQ, TSLA)", value="QQQ").upper()

if st.sidebar.button("Refresh Data"):
    st.rerun()

# Disclaimer for Play Store Compliance
st.sidebar.info("⚠️ Not financial advice. For educational purposes only.")

# --- MAIN LOGIC ---
with st.spinner(f"Analyzing {symbol}..."):
    raw_data = fetch_data(symbol)
    
    if raw_data is not None:
        processed_data = calculate_indicators(raw_data)
        
        if len(processed_data) > 20:
            current_price = processed_data['Close'].iloc[-1]
            pred_price = predict_price(processed_data)
            change = pred_price - current_price
            
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("Predicted (5m)", f"${pred_price:.2f}", f"{change:.2f}")
            col3.metric("RSI", f"{processed_data['RSI'].iloc[-1]:.1f}")

            # Chart
            st.line_chart(processed_data['Close'])
            
            st.success("AI Model Updated Successfully")
        else:
            st.warning("Waiting for more market data to build prediction...")
    else:
        st.error("Market data unavailable. Check the ticker or market hours.")

# Auto-refresh every 30 seconds
time.sleep(30)
st.rerun()
