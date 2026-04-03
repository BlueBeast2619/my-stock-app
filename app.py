def build_model(symbol, window):
    # 1. Download and immediately fix the multi-column issue
    df = yf.download(symbol, period="5d", interval="1m", prepost=True, progress=False)
    
    if df.empty:
        return None, None

    # This is the 'Magic Line' that fixes the ValueError:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Technical Indicators (Manual Calculations)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['Volatility'] = df['Close'].rolling(window=15).std()
    df['Momentum'] = df['Close'].diff(10)
    
    # 3. Volume Pressure (This will now work without error!)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Pressure'] = df['Volume'] / df['Vol_Avg']
    
    # 4. Target & Model Training
    df['Target'] = df['Close'].shift(-window)
    features = ['Close', 'RSI', 'SMA_9', 'Volatility', 'Momentum', 'Vol_Pressure']
    data = df.dropna()
    
    if data.empty:
        return None, df

    model = Ridge(alpha=1.0)
    model.fit(data[features], data['Target'])
    
    # 5. Extract Final Metrics
    latest_row = data[features].tail(1)
    
    metrics = {
        "price": float(df['Close'].values.flatten()[-1]),
        "vol_p": float(df['Vol_Pressure'].values.flatten()[-1]),
        "momentum": float(df['Momentum'].values.flatten()[-1]),
        "prediction": float(model.predict(latest_row).flatten()[0])
    }
    
    return metrics, df
