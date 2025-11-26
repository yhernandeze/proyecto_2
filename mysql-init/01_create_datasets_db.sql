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

    -- Example feature columns (we’ll refine once we see the exact schema)
    beds INT,
    baths INT,
    sqft_living INT,
    sqft_lot INT,
    floors FLOAT,
    waterfront TINYINT(1),
    view INT,
    condition INT,
    grade INT,
    sqft_above INT,
    sqft_basement INT,
    yr_built INT,
    yr_renovated INT,
    lat DOUBLE,
    long DOUBLE,
    sqft_living15 INT,
    sqft_lot15 INT,
    city VARCHAR(100),
    state VARCHAR(50),
    zipcode VARCHAR(20),

    -- Target
    price DOUBLE
);
