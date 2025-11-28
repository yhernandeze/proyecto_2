-- Create a database to store the project datasets (if not exists)
CREATE DATABASE IF NOT EXISTS datasets_db;
USE datasets_db;

-- RAW table: mirror incoming JSON from house prices API
CREATE TABLE IF NOT EXISTS raw_house_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    batch_id INT NOT NULL,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_payload JSON NOT NULL
);

-- CLEAN table: processed features + target price
CREATE TABLE IF NOT EXISTS clean_house_prices (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    batch_id INT NOT NULL,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Metadata from the API
    group_number INT,
    api_day VARCHAR(20),
    api_batch_number INT,

    -- Feature columns aligned with the model
    brokered_by VARCHAR(255),
    status VARCHAR(50),
    price DOUBLE,
    bed DOUBLE,
    bath DOUBLE,
    acre_lot DOUBLE,
    street DOUBLE,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code DOUBLE,
    house_size DOUBLE,
    prev_sold_date VARCHAR(50)
);

-- Table to register training runs and decisions
CREATE TABLE IF NOT EXISTS training_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id VARCHAR(64),
    last_seen_batch_id INT,
    total_rows INT,
    new_rows INT,
    new_ratio DOUBLE,
    mean_price DOUBLE,
    var_price DOUBLE,
    drift_mean DOUBLE,
    drift_var DOUBLE,
    drift_score DOUBLE,
    retrain_decision VARCHAR(50),
    promotion_decision VARCHAR(50),
    decision_reason TEXT,
    promoted_to_production TINYINT(1),
    training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to log inference calls (used by the API)
CREATE TABLE IF NOT EXISTS inference_requests (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    run_id VARCHAR(64),
    bed DOUBLE,
    bath DOUBLE,
    acre_lot DOUBLE,
    house_size DOUBLE,
    zip_code DOUBLE,
    brokered_by DOUBLE,
    street DOUBLE,
    predicted_price DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
