import os
import time
import datetime
import warnings
from dotenv import load_dotenv

# from arima_forecast import ARIMAForecast
from olhc_forecast import OLHCForecast

warnings.filterwarnings("ignore")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", 3600))  # default: 1 hour

def wait_until_next_run():
    """Sleep until the next 0, 4, 8, 12, 16, 20-hour mark."""
    now = datetime.datetime.now()
    next_hour = ((now.hour // 4) + 1) * 4
    if next_hour >= 24:
        # move to next day 00:00
        next_run = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        next_run = now.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    
    sleep_seconds = (next_run - now).total_seconds()
    print(f"[INFO] Next run scheduled at {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({sleep_seconds/3600:.2f} hours from now)")
    time.sleep(sleep_seconds)


if __name__ == "__main__":
    forecast = OLHCForecast()
    while True:
        start = datetime.datetime.now()
        print(f"[INFO] Starting forecast run at {start.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            forecast.run()
        except Exception as e:
            print(f"[ERROR] Forecast run failed: {e}")
        wait_until_next_run()
        