-- Base para MLflow (ya se crea por variables, pero garantizamos)
CREATE DATABASE IF NOT EXISTS mlflow_db;

-- Base para datos de proyecto (RAW/CURATED)
CREATE DATABASE IF NOT EXISTS datasets_db;

-- Usuario con permisos (si no existiese)
CREATE USER IF NOT EXISTS 'mlflow_user'@'%' IDENTIFIED BY 'mlflow_pass';
GRANT ALL PRIVILEGES ON mlflow_db.* TO 'mlflow_user'@'%';
GRANT ALL PRIVILEGES ON datasets_db.* TO 'mlflow_user'@'%';
FLUSH PRIVILEGES;
 