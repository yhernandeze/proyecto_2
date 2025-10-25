FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y gcc python3-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY main.py .

# Crear directorio para modelos
RUN mkdir -p /app/models

# Exponer puerto
EXPOSE 8989

# Healthcheck
#HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    #CMD curl -f http://localhost:8989/health || exit 1

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]