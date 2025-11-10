FROM python:3.11-slim

WORKDIR /app
COPY requirements-locust.txt .
RUN pip install --no-cache-dir -r requirements-locust.txt

# Copy test script
COPY locustfile.py /app/locustfile.py

# Locust listens on 8089; Prometheus endpoint on 9649 when enabled
EXPOSE 8089 9649

# Allow passing locust options via env (LOCUST_OPTS)
ENV LOCUST_OPTS=""

ENTRYPOINT ["sh", "-c", "locust -f /app/locustfile.py $LOCUST_OPTS"]