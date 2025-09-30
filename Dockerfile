FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/lightfm

COPY . .

RUN pip install --no-cache-dir . pytest scikit-learn && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    cp -r tests /home/tests && \
    rm -rf /home/lightfm

WORKDIR /home
