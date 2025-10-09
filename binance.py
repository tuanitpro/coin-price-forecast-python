import os
import time
import polars as pl
import requests

class Binance:
    def __init__(self):
        self.interval = os.getenv("INTERVAL", "4h")
        self.limit = int(os.getenv("LIMIT", 200))
        if not self.interval or not self.limit:
            print("[WARN] Binance interval or limit is missing in .env")

    def fetch(self, symbol: str) -> pl.DataFrame:
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
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
        ]

        # Build DataFrame with explicit schema
        df = pl.DataFrame(data, schema=cols)

        # Convert timestamp and numeric columns
        df = df.with_columns([
            pl.col("open_time").cast(pl.Int64),
            pl.col("close_time").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64)
        ])

        # Add datetime column
        df = df.with_columns(
            pl.from_epoch("open_time", time_unit="ms").alias("date")
        )

        # Sort by date ascending
        df = df.sort("date")

        # Set "date" as index (for pandas compatibility)
        df = df.rename({"date": "index"}).with_columns(pl.col("index"))
        df = df.rename({"index": "date"})

        return df
