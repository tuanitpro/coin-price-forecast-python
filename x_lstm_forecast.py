import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from telegram_notifier import TelegramNotifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "DOTUSDT,ADAUSDT").split(",")]
STEPS = int(os.getenv("STEPS", 5))
INTERVAL = os.getenv("INTERVAL", "1d")
LIMIT = int(os.getenv("LIMIT", 500))
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", 3600))
THRESHOLD_BUY = float(os.getenv("THRESHOLD_BUY", 2))
THRESHOLD_SELL = float(os.getenv("THRESHOLD_SELL", -2))
SEQ_LEN = 30

# -----------------------------
# Functions
# -----------------------------
def fetch_binance(symbol, interval=INTERVAL, limit=LIMIT):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","taker_base","taker_quote","ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date","close"]].set_index("date")
    df = df.asfreq("D").interpolate()
    return df

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_future(model, last_seq, scaler, steps=STEPS):
    predictions = []
    seq = last_seq.copy()
    for _ in range(steps):
        pred = model.predict(seq.reshape(1, seq.shape[0], 1), verbose=0)
        predictions.append(pred[0,0])
        seq = np.append(seq[1:], pred[0,0])
    predictions = np.array(predictions).reshape(-1,1)
    return scaler.inverse_transform(predictions).flatten()

# -----------------------------
# Main Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        notifier = TelegramNotifier()
        for symbol in SYMBOLS:
            print(f"\n[INFO] Forecasting {symbol} at {pd.Timestamp.now()}")
            try:
                df = fetch_binance(symbol)
            except Exception as e:
                print(f"[ERROR] Failed to fetch {symbol}: {e}")
                continue

            if len(df) < SEQ_LEN:
                print(f"[WARN] Not enough data for {symbol}")
                continue

            X, y, scaler = prepare_data(df)
            model = build_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=10, batch_size=16, verbose=0)

            last_seq = X[-1,:,0]
            future_prices = predict_future(model, last_seq, scaler, STEPS)

            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=STEPS)
            forecast_df = pd.DataFrame({"date": future_dates, "predicted_price": future_prices})
            forecast_df["pct_change"] = forecast_df["predicted_price"].pct_change().mul(100)

            print("[INFO] Top 5 forecasted prices:")
            print(forecast_df.head())

            # Build Telegram message with labels
            message_lines = [f"*LSTM Forecast for {symbol}*"]
            for idx, row in forecast_df.head(5).iterrows():
                pct = row['pct_change']
                if pct > THRESHOLD_BUY:
                    label = "Buy 🟢"
                elif pct < THRESHOLD_SELL:
                    label = "Sell 🔴"
                else:
                    label = "Hold ⚪"
                line = f"{row['date'].strftime('%Y-%m-%d %H:%M:%S')}: {row['predicted_price']:.4f} ({pct:+.2f}%) - {label}"
                message_lines.append(line)

            message = "\n".join(message_lines)
            notifier.send(message)

        print(f"[INFO] Sleeping for {SLEEP_SECONDS} seconds...\n")
        time.sleep(SLEEP_SECONDS)
