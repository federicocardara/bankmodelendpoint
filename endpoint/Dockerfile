FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    libgomp1 \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*


COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

COPY model_endpoint.py /app/model_endpoint.py

CMD ["python", "model_endpoint.py"]
