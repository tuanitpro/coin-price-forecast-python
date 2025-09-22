FROM python:3.13.7-alpine3.22

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./x_arima_forecast.py" ]