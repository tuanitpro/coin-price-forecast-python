import os
import requests

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_TO")

        if not self.token or not self.chat_id:
            print("[WARN] Telegram token or chat ID is missing in .env")

    def send(self, message: str):
        if not self.token or not self.chat_id:
            return  # skip sending if not configured

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}

        try:
            r = requests.post(url, data=payload, timeout=10)
            r.raise_for_status()
            print("✅ [INFO] Telegram message sent successfully")
        except Exception as e:
            print(f"🚫 [ERROR] Failed to send Telegram message: {e}")
