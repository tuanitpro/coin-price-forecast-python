import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

SYMBOL = os.getenv("SYMBOL", "TRUMPUSDT")
INTERVAL = os.getenv("INTERVAL", "1h")  # or "1d"
LIMIT = int(os.getenv("LIMIT", 1000))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_TO = os.getenv("TELEGRAM_TO")
THRESHOLD = float(os.getenv("PRICE_CHANGE_THRESHOLD", 5.0))   # percent

def fetch_klines(symbol, interval, limit, max_retries=3):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"[WARN] fetch attempt {attempt+1} failed: {e}")
            time.sleep(2)
    else:
        raise RuntimeError("Failed to fetch klines after retries")

    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date")
    df = df.set_index("date")
    return df

def build_features(df):
    # example features: open-close, high-low, volume, moving averages
    df2 = df.copy()
    df2["oc_diff"] = df2["open"] - df2["close"]
    df2["hl_diff"] = df2["high"] - df2["low"]
    df2["ma_5"] = df2["close"].rolling(window=5).mean()
    df2["ma_10"] = df2["close"].rolling(window=10).mean()
    df2["volume_change"] = df2["volume"].pct_change()
    df2 = df2.dropna()
    return df2

def prepare_training_data(df_feat):
    # target: next period close price
    df_feat["target"] = df_feat["close"].shift(-1)
    df_feat = df_feat.dropna()
    X = df_feat.drop(columns=["target"])
    y = df_feat["target"]
    
    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, X.columns.tolist()

def send_telegram_message(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_TO:
        print("[WARN] Telegram token or chat id not set")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_TO, "text": msg, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        print("[INFO] Telegram message sent")
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

def main():
    print(f"[INFO] Running ML price forecast {SYMBOL} at {datetime.now(timezone.utc)}")
    df = fetch_klines(SYMBOL, INTERVAL, LIMIT)
    df_feat = build_features(df)
    
    X, y, scaler, feature_cols = prepare_training_data(df_feat)
    if len(X) < 50:
        print("[WARN] Not enough data")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # forecast next close price
    last_features = df_feat.iloc[-1:].drop(columns=["target"], errors="ignore")
    X_last = scaler.transform(last_features[feature_cols])
    predicted_price = model.predict(X_last)[0]

    current_price = df_feat["close"].iloc[-1]
    change_pct = (predicted_price - current_price) / current_price * 100

    # build message
    msg = (
        f"*ML Forecast for {SYMBOL}*\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Current Price: {current_price:.4f}\n"
        f"Predicted Next Close: {predicted_price:.4f}\n"
        f"Change: {change_pct:+.2f}%\n"
    )

    # decide signal
    if change_pct > THRESHOLD:
        msg += "Signal: *Buy 🟢*"
    elif change_pct < -THRESHOLD:
        msg += "Signal: *Sell 🔴*"
    else:
        msg += "Signal: *Hold ⚪*"

    send_telegram_message(msg)

if __name__ == "__main__":
    main()
