import streamlit as st
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import os

# Streamlit page setup
st.set_page_config(page_title="Stock LSTM Predictor", layout="wide")
st.title("📈 LSTM Stock Price Forecast")

# Sidebar inputs
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
n_days_forecast = st.sidebar.slider("Days to Forecast", 1, 30, 7)
retrain = st.sidebar.checkbox("Force Retrain Model", value=False)
run = st.sidebar.button("Predict")

@st.cache_data(show_spinner=False)
def fetch_data(symbol, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='full')
    df = data.rename(columns={'4. close': 'Close'})
    df = df[['Close']].sort_index()
    return df.dropna()

def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def forecast(model, last_seq, days, scaler):
    predictions = []
    input_seq = last_seq.copy()
    for _ in range(days):
        pred = model.predict(input_seq.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0][0])
        input_seq = np.append(input_seq[1:], pred)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

if run:
    try:
        df = fetch_data(symbol, api_key)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']])

        window_size = 60
        X, y = create_dataset(scaled, window_size)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model_path = f"{symbol}_lstm_model.h5"

        if os.path.exists(model_path) and not retrain:
            model = load_model(model_path)
            st.success("Loaded existing model.")
        else:
            st.info("Training new model...")
            model = train_lstm_model(X, y)
            model.save(model_path)
            st.success("Model trained and saved.")

        pred = model.predict(X)
        pred_prices = scaler.inverse_transform(pred)
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

        # Plot predictions
        st.subheader("📊 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[window_size:], actual_prices, label="Actual")
        ax.plot(df.index[window_size:], pred_prices, label="Predicted")
        ax.set_title(f"{symbol} Stock Price Prediction")
        ax.legend()
        st.pyplot(fig)

        # Forecast future
        last_window = scaled[-window_size:]
        future_preds = forecast(model, last_window, n_days_forecast, scaler)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days_forecast, freq='B')

        df_forecast = pd.DataFrame({'Forecasted Close': future_preds.flatten()}, index=future_dates)
        st.subheader(f"🔮 Next {n_days_forecast} Days Forecast")
        st.line_chart(df_forecast)

        # Download CSV
        all_preds = pd.concat([df.iloc[window_size:].copy(), pd.DataFrame(pred_prices, index=df.index[window_size:], columns=["Predicted Close"])], axis=1)
        output = pd.concat([all_preds, df_forecast], axis=0)
        csv = output.to_csv().encode()
        st.download_button("📥 Download CSV", csv, file_name=f"{symbol}_forecast.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
