FROM python:3.13 AS builder
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apk add --no-cache musl-dev gcc g++ make
RUN pip install pyinstaller

COPY . .
RUN pyinstaller --onefile main.py

FROM gcr.io/distroless/static-debian12
WORKDIR /bin

COPY --from=builder /usr/src/app/dist/main .
CMD ["./main"]
