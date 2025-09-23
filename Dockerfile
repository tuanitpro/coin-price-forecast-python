FROM python:3.13.7-alpine3.22
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /usr/src/app

RUN apk add --no-cache --virtual .build-deps \
      g++ gcc gfortran musl-dev lapack-dev \
      libstdc++ openblas-dev
      
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./main.py" ]