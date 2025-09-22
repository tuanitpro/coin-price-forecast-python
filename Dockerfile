# Builder stage
FROM python:3.13.7-alpine3.22 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Final stage
FROM gcr.io/distroless/static-debian12
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/python3.13 /usr/local/bin/python3.13
COPY --from=builder /app /app
WORKDIR /app
CMD ["/usr/local/bin/python3.13", "x_arima_forecast.py"]