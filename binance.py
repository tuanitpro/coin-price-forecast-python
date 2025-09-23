
import os
import time
import pandas as pd
import requests

class Binance: 
    def __init__(self):
        self.interval =  os.getenv("INTERVAL", "1d")
        self.limit =  int(os.getenv("LIMIT", 1000))
        if not self.interval or not self.limit:
            print("[WARN] Binance interval or limit is missing in .env")
    
    def fetch(self, symbol: str):
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": self.interval, "limit": self.limit}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                print(f"[INFO] Fetched {len(data)} klines for {symbol}")
                break
            except Exception as e:
                print(f"[WARN] Attempt {attempt+1} failed for {symbol}: {e}")
                if attempt + 1 == max_retries:
                    raise
                time.sleep(2)

        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_values("date")
        df = df.set_index("date")
        return df