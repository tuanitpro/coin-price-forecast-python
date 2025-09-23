import os
import time
import warnings
from dotenv import load_dotenv

from arima_forecast import ARIMAForecast

warnings.filterwarnings("ignore")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
SLEEP_SECONDS = int(os.getenv("SLEEP_SECONDS", 3600))  # default: 1 hour

# -----------------------------
# Main loop
# -----------------------------

if __name__ == "__main__":
    forecast = ARIMAForecast()
    while True:
        forecast.run()
        print(f"[INFO] Sleeping for {SLEEP_SECONDS} seconds...\n")
        time.sleep(SLEEP_SECONDS)
        