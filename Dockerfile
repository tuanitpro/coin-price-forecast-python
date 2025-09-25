FROM python:3.13-alpine AS builder
RUN apk add --no-cache musl-dev gcc g++ make
RUN pip install --no-cache-dir pyinstaller

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pyinstaller --onefile main.py

# runtime stage
FROM gcr.io/distroless/cc-debian12

WORKDIR /bin
COPY --from=builder /app/dist/main .
CMD ["./main"]
