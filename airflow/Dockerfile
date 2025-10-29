FROM apache/airflow:2.7.3-python3.10

USER root
RUN apt-get update && \
    apt-get install -y gcc python3-dev curl && \
    apt-get clean

USER airflow


RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3==1.34.12 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.2 \
    requests==2.31.0 \
    pymysql==1.1.0 \
    ucimlrepo==0.0.7

    RUN mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins
WORKDIR /opt/airflow