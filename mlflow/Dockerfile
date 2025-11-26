FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y gcc python3-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Instalar MLflow y dependencias
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3==1.34.12 \
    pymysql==1.1.0 \
    cryptography==41.0.7 \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2

# Puerto para MLflow
EXPOSE 5000

# Healthcheck (use root, MLflow has no /health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000 || exit 1