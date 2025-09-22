import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from statsmodels.tsa.statespace.sarimax import SARIMAX

from binance import Binance
from telegram import TelegramNotifier

class ARIMAForecast:
    def __init__(self):
        self.steps = int(os.getenv("STEPS", 30))
        self.symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "DOTUSDT").split(",")]
        self.threshold_buy = float(os.getenv("THRESHOLD_BUY", 2))
        self.threshold_sell = float(os.getenv("THRESHOLD_SELL", -2))
        self.previous_results = {}

        if not self.symbols:
            print("[WARN] The symbols is missing in .env")


    def _auto_arima_grid_search(self, y, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
        best_aic = np.inf
        best_order = None
        best_model = None
        for d in d_range:
            for p in range(p_range[0], p_range[1]+1):
                for q in range(q_range[0], q_range[1]+1):
                    try:
                        model = SARIMAX(
                            y, order=(p,d,q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        res = model.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p,d,q)
                            best_model = res
                    except:
                        continue
        print(f"[INFO] Selected ARIMA order={best_order} with AIC={best_aic:.2f}")
        return best_model, best_order


    def _run_auto_arima_forecast(self, df):
        y = np.log(df["price"])
        model, order = self._auto_arima_grid_search(y)
        if model is None:
            raise ValueError("[ERROR] No valid ARIMA model found during grid search.")
        forecast = model.get_forecast(steps=self.steps)
        pred_mean = np.exp(forecast.predicted_mean)
        ci = np.exp(forecast.conf_int())
        freq = pd.infer_freq(df.index) or "D"
        future_index = pd.date_range(df.index[-1] + pd.tseries.frequencies.to_offset(freq),
                                    periods=self.steps, freq=freq)
        fc = pd.DataFrame({
            "date": future_index,
            "predicted_price": pred_mean.values,
            "lower_ci": ci.iloc[:,0].values,
            "upper_ci": ci.iloc[:,1].values
        })
        fc["pct_change"] = fc["predicted_price"].pct_change().mul(100)
        fc["date"] = pd.to_datetime(fc["date"], utc=True)
        return fc

    def run(self):
        now = datetime.now(timezone.utc)
        notifier = TelegramNotifier()
        binance = Binance()
        for symbol in self.symbols:
            print(f"\n[INFO] Starting forecast for {symbol} at {pd.Timestamp.now()}")
            df = binance.fetch(symbol)

            current_price = df["price"].iloc[-1]
            prev_price = self.previous_results.get(symbol)
            percent_change = None
            if prev_price is not None:
                percent_change = ((current_price - prev_price) / prev_price) * 100
            self.previous_results[symbol] = current_price
            
            print(f"📢 📢 📢 [INFO] Last actual price: {current_price:.4f}")
            forecast = self._run_auto_arima_forecast(df)
            forecast = forecast[forecast["date"] > now]

            top5 = forecast.head(5)
            print("\n[INFO] Top 5 forecasted prices:")
            print(top5[['date','predicted_price','pct_change']])

            # Build Telegram message with full timestamp
            message_lines = [f"📢 📢 📢 *ARIMA Forecast for {symbol}*"]
            if percent_change is not None:
                message_lines.append(f"Last actual price: {current_price:.4f} 🔥 Change: {percent_change:.2f}%")
            else:
                message_lines.append(f"Last actual price: {current_price:.4f}")
            
            for idx, row in forecast.head(5).iterrows():
                pct = row['pct_change']
                if pct > self.threshold_buy:
                    label = "Buy ✅"
                elif pct < self.threshold_sell:
                    label = "Sell 🎯"
                else:
                    label = "Hold 🚫"
                line = f"{row['date'].strftime('%d/%m/%y %H:%M')}: {row['predicted_price']:.4f} ({pct:+.2f}%) - {label}"
                message_lines.append(line)
            message = "\n".join(message_lines)

            notifier.send(message)
