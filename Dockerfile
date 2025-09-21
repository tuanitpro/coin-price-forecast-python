FROM python:3.13.7-alpine3.22

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV SYMBOL=DOTUSDT STEPS=30 INTERVAL=1d LIMIT=1000 SLEEP_SECONDS=3600
ENV TELEGRAM_TOKEN= TELEGRAM_TO=
CMD [ "python", "./x_forecast.py" ]
