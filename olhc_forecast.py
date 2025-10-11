import os
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error

from binance import Binance
from telegram import TelegramNotifier

class OLHCForecast:
    def __init__(self):
        self.interval = os.getenv("INTERVAL", "4h")
        self.limit = int(os.getenv("LIMIT", 200))
        self.steps = int(os.getenv("STEPS", 5))
        self.symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT").split(",")]
        self.threshold = float(os.getenv("PRICE_CHANGE_THRESHOLD", 2))
        self.previous_results = {}

        if not self.symbols:
            print("[WARN] The symbols are missing in .env")

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    def _build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            (df["open"] - df["close"]).alias("oc_diff"),
            (df["high"] - df["low"]).alias("hl_diff"),
            df["close"].rolling_mean(5).alias("ma_5"),
            df["close"].rolling_mean(10).alias("ma_10"),
            df["volume"].pct_change().alias("volume_change"),
        ])
        df = self._add_indicators(df)
        df = df.drop_nulls()
        return df

    def _add_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        # --- MACD ---
        ema12 = df["close"].ewm_mean(span=12)
        ema26 = df["close"].ewm_mean(span=26)
        macd = ema12 - ema26
        signal = macd.ewm_mean(span=9)
        macd_hist = macd - signal

        df = df.with_columns([
            ema12.alias("ema12"),
            ema26.alias("ema26"),
            macd.alias("macd"),
            signal.alias("signal"),
            macd_hist.alias("macd_hist")
        ])

        # --- RSI + StochRSI ---
        diff = df["close"].diff()
        gain = diff.map_elements(lambda x: x if x > 0 else 0.0)
        loss = diff.map_elements(lambda x: -x if x < 0 else 0.0)

        roll_up = gain.rolling_mean(14)
        roll_down = loss.rolling_mean(14)
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))

        rsi_min = rsi.rolling_min(14)
        rsi_max = rsi.rolling_max(14)
        stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min)

        df = df.with_columns([
            rsi.alias("rsi"),
            stochrsi.alias("stochrsi"),
        ])

        df = df.drop_nulls()
        return df

    # -----------------------------
    # Data Preparation
    # -----------------------------
    def _prepare_training_data(self, df_feat: pl.DataFrame):
        df_feat = df_feat.with_columns(
            df_feat["close"].shift(-1).alias("target")
        ).drop_nulls()

        # Drop any datetime or non-numeric columns before training
        numeric_cols = [
            c for c, dtype in zip(df_feat.columns, df_feat.dtypes)
            if dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
        ]

        if "target" in numeric_cols:
            numeric_cols.remove("target")

        X = df_feat.select(numeric_cols)
        y = df_feat["target"].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.to_numpy())
        return X_scaled, y, scaler, numeric_cols


    # -----------------------------
    # Multi-step Forecast
    # -----------------------------
    def _multi_step_forecast(self, model, last_features, scaler, feature_cols, steps=5):
        preds = []
        current_features = last_features.copy()
        for _ in range(steps):
            X_last = scaler.transform(current_features[feature_cols])
            pred = model.predict(X_last)[0]
            preds.append(pred)
            # Simulate next step input
            current_features["close"] = pred
        return preds

    # -----------------------------
    # Main Runner
    # -----------------------------
    def run(self):
        notifier = TelegramNotifier()
        binance = Binance()

        for symbol in self.symbols:
            print(f"\n[INFO] Starting forecast for {symbol} ...")
            df = binance.klines(symbol, interval=self.interval, limit=self.limit)
            if df.height < 50:
                print(f"[WARN] Not enough data for {symbol}")
                continue

            df_feat = self._build_features(df)
            X, y, scaler, feature_cols = self._prepare_training_data(df_feat)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            #rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Forecast future prices
            last_features = df_feat.tail(1).to_pandas()
            preds = self._multi_step_forecast(model, last_features, scaler, feature_cols, self.steps)

            current_price = df_feat["close"].to_list()[-1]
            next_price = preds[0]
            change_pct = (next_price - current_price) / current_price * 100
            klines = binance.klines(symbol, "1d", 1)
            
            # Build message
            msg = (
                f"📢 📢 📢 *ML Forecast for #{symbol}*\n"
                f"Change: {change_pct:+.2f}% ({current_price:.4f} → {next_price:.4f})\n"
            )

            if change_pct > self.threshold:
                msg += "Signal: *Buy ✅*\n"
            elif change_pct < -self.threshold:
                msg += "Signal: *Sell 🎯*\n"
            else:
                msg += "Signal: *Hold 🚫*\n"
            msg +=   f"High: {klines["high"][0]:.4f}\n"
            msg +=   f"Low: {klines["low"][0]:.4f}\n"
            msg +=   f"Current Price: {current_price:.4f}\n"
            msg +=   f"Next Price: {next_price:.4f}\n"
            #msg += f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            #msg += "\nNext prices:\n" + ", ".join([f"{p:.4f}" for p in preds])

            print(msg)
            notifier.send(msg)


# if __name__ == "__main__":
#     OLHCForecast().run()
