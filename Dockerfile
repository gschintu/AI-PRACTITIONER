FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "train_mnist.py", "--epochs", "1", "--batch-size", "128", "--lr", "0.02"]
