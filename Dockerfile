# ---------- Stage 1: Build binary ----------
FROM python:3.13-slim AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyinstaller

# Copy source
COPY . .

# Build standalone binary
RUN pyinstaller --onefile main.py --name app --clean --noconfirm

# ---------- Stage 2: Minimal runtime ----------
FROM alpine:3.20 AS runner

# Add required runtime libraries
RUN apk add --no-cache libstdc++ libgcc

WORKDIR /app
COPY --from=builder /app/dist/app .

# Final entrypoint
CMD ["./app"]
