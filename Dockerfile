# Stage 1: Build dependencies
FROM python:3.13-slim-buster AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final, minimal image
FROM python:3.13-slim-buster

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY . .

CMD ["python", "x_arima_forecast.py"]