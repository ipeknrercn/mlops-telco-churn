FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY data/ ./data/
CMD ["python", "scripts/train_cleaned_data.py"]
