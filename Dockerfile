FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir pyinstaller && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pyinstaller --onefile main.py --name app --clean --noconfirm

FROM debian:stable-slim
WORKDIR /app
COPY --from=builder /app/dist/app .
CMD ["./app"]
