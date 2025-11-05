FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y gcc python3-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.5.3 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    sqlalchemy==2.0.23 \
    pymysql==1.1.0 \
    ucimlrepo==0.0.7 \
    requests==2.31.0

COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8080"]
 
 