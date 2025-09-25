FROM python:3.13-alpine as builder
RUN apk add --no-cache musl-dev gcc g++ make
RUN pip install pyinstaller

WORKDIR /app
COPY . .
RUN pyinstaller --onefile main.py

FROM scratch
COPY --from=builder /app/dist/main /
CMD ["./main"]
