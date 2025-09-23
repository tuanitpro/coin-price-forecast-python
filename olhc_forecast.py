import os
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from binance import Binance
from telegram import TelegramNotifier

class OLHCForecast:
    def __init__(self):
        self.steps = int(os.getenv("STEPS", 30))
        self.symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "DOTUSDT").split(",")]
        self.threshold = float(os.getenv("PRICE_CHANGE_THRESHOLD", 2))
        self.previous_results = {}

        if not self.symbols:
            print("[WARN] The symbols is missing in .env")

    def _build_features(self, df):
        # example features: open-close, high-low, volume, moving averages
        df2 = df.copy()
        df2["oc_diff"] = df2["open"] - df2["close"]
        df2["hl_diff"] = df2["high"] - df2["low"]
        df2["ma_5"] = df2["close"].rolling(window=5).mean()
        df2["ma_10"] = df2["close"].rolling(window=10).mean()
        df2["volume_change"] = df2["volume"].pct_change()
        df2 = df2.dropna()
        return df2

    def _prepare_training_data(self, df_feat):
        # target: next period close price
        df_feat["target"] = df_feat["close"].shift(-1)
        df_feat = df_feat.dropna()
        X = df_feat.drop(columns=["target"])
        y = df_feat["target"]
        
        # scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, X.columns.tolist()

    def run(self):
        notifier = TelegramNotifier()
        binance = Binance()
        for symbol in self.symbols:
            print(f"📢 📢 📢 [INFO] Running ML price forecast {symbol} at {datetime.now(timezone.utc)}")
            df = binance.fetch(symbol)
            df_feat = self._build_features(df)
            
            X, y, scaler, feature_cols = self._prepare_training_data(df_feat)
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
                f"📢 📢 📢 *ML Forecast for {symbol}*\n"
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Current Price: {current_price:.4f}\n"
                f"Predicted Next Close: {predicted_price:.4f}\n"
                f"Change: {change_pct:+.2f}%\n"
            )
            # decide signal
            if change_pct > self.threshold:
                msg += "Signal: *Buy ✅*"
            elif change_pct < -self.threshold:
                msg += "Signal: *Sell 🎯*"
            else:
                msg += "Signal: *Hold 🚫*"
            print(msg)
            notifier.send(msg)
