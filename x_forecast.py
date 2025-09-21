import os
import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import time
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

SYMBOL = os.getenv("SYMBOL", "DOTUSDT").upper()
STEPS = int(os.getenv("STEPS", 30))
INTERVAL = os.getenv("INTERVAL", "1d")
LIMIT = int(os.getenv("LIMIT", 1000))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_TO = os.getenv("TELEGRAM_TO")


# -----------------------------
# 1) Fetch OHLCV data
# -----------------------------
def fetch_binance(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT, max_retries=3):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            print(f"[INFO] Successfully fetched {len(data)} klines for {symbol}")
            break
        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed: {e}")
            if attempt + 1 == max_retries:
                raise
            time.sleep(2)

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = pd.to_numeric(df["close"], errors="coerce")

    df = df[["date", "price"]].set_index("date")
    df = df.asfreq(pd.infer_freq(df.index) or "D")
    df["price"] = df["price"].interpolate()

    print(f"[INFO] Dataframe shape after interpolation: {df.shape}")
    return df


# -----------------------------
# 2) Grid Search ARIMA
# -----------------------------
def auto_arima_grid_search(y, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    best_aic = np.inf
    best_order = None
    best_model = None

    # print("[INFO] Starting ARIMA grid search...")
    # total_tests = len(p_range) * len(d_range) * len(q_range)
    test_count = 0

    for d in d_range:
        for p in range(p_range[0], p_range[1] + 1):
            for q in range(q_range[0], q_range[1] + 1):
                test_count += 1
                try:
                    model = SARIMAX(
                        y, order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = model.fit(disp=False)
                    # print(f"[GRID] Tested ARIMA({p},{d},{q}) AIC={res.aic:.2f} ({test_count}/{total_tests})")
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_model = res
                except:
                    continue
    # print(f"[INFO] Selected ARIMA order={best_order} with AIC={best_aic:.2f}")
    return best_model, best_order


# -----------------------------
# 3) Forecast function
# -----------------------------
def run_auto_arima_forecast(df, steps=STEPS):
    y = np.log(df["price"])

    model, order = auto_arima_grid_search(y)

    forecast = model.get_forecast(steps=steps)
    pred_mean = np.exp(forecast.predicted_mean)
    ci = np.exp(forecast.conf_int())

    freq = pd.infer_freq(df.index) or "D"
    future_index = pd.date_range(df.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                 periods=steps, freq=freq)

    fc = pd.DataFrame({
        "date": future_index,
        "predicted_price": pred_mean.values,
        "lower_ci": ci.iloc[:, 0].values,
        "upper_ci": ci.iloc[:, 1].values
    })
    fc["pct_change"] = fc["predicted_price"].pct_change().mul(100)

    print(f"[INFO] Forecast generated for {steps} steps")
    return fc


# -----------------------------
# 4) Send Telegram message
# -----------------------------
def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_TO:
        print("[WARN] Telegram token or chat ID not set in .env")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_TO, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
        print("[INFO] Telegram message sent successfully")
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram message: {e}")


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    df = fetch_binance(SYMBOL, INTERVAL, LIMIT)
    print(f"[INFO] Last actual price for {SYMBOL}: {df['price'].iloc[-1]:.4f}")

    forecast = run_auto_arima_forecast(df, steps=STEPS)

    # Take top 5 forecasted results
    top5 = forecast.head(5)

    # Print to console
    print("\n[INFO] Top 5 forecasted prices:")
    print(top5[['date', 'predicted_price', 'pct_change']])

    # Prepare Telegram message
    message_lines = [f"*ARIMA Forecast for {SYMBOL}*"]
    for idx, row in top5.iterrows():
        line = f"{row['date'].strftime('%d/%m %H:%M')}: {row['predicted_price']:.4f} ({row['pct_change']:+.2f}%)"
        message_lines.append(line)
    message = "\n".join(message_lines)

    send_telegram_message(message)
    print("[INFO] Script completed successfully.")
