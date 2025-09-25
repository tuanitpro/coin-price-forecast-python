FROM python:3.12-alpine as builder
RUN apk add --no-cache musl-dev gcc g++ make
RUN pip install pyinstaller

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pyinstaller --onefile main.py

FROM gcr.io/distroless/static-debian12
WORKDIR /bin

COPY --from=builder /app/dist/main .
CMD ["./main"]
