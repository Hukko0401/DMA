# DMA — Vinhomes Real Estate Price Prediction

Predicting real estate listing prices for Vinhomes properties using web-scraped market data, with a focus on identifying the key price drivers for Vinhomes-branded listings versus competitors.

## Overview

- Scraped **4,362 real estate listings** from batdongsan.com and related sources
- Combined structured features (area, distance to CBD/metro/river, amenities...) with **TF-IDF text embeddings** from Vietnamese listing descriptions
- Trained and compared multiple regression models (Linear Regression, Random Forest, XGBoost), tuned via **RandomizedSearchCV**
- Best model: **XGBoost** → **R² = 0.9617** on the test set
- Applied **SHAP** analysis to interpret feature importance and identify `brand_Vinhomes` as the top price driver
- Applied **K-means clustering (k=3)** to segment the market and benchmark Vinhomes' competitive positioning
- Deployed a prediction AI agent via **n8n + Render + Supabase**

## Pipeline

The project is organized into 3 notebooks, reflecting the full workflow:

1. **`DMA_Normalize_Clean.ipynb`** — Data loading, schema validation, and cleaning (type conversion, handling of numeric fields such as distance-to-CBD, distance-to-metro, land area, construction density)
2. **`DMA_EDA.ipynb`** — Exploratory data analysis and competitive benchmarking, including digital marketplace presence analysis and brand listing share for Vinhomes vs. competitors
3. **`DMA_TrainModel_ML.ipynb`** — Feature engineering (including TF-IDF on listing descriptions), model training and comparison (Linear/RandomForest/XGBoost), hyperparameter tuning, and SHAP-based interpretation

`app.py` contains the deployed inference service (Flask), with the trained model and preprocessing artifacts (`model.pkl`, `scaler.pkl`, `tfidf.pkl`, `columns.pkl`).

## Tech Stack

Python · pandas · numpy · scikit-learn · XGBoost · SHAP · TF-IDF (scikit-learn) · Flask · Supabase · n8n

## Key Results

| Metric | Value |
|---|---|
| R² (test set) | 0.9617 |
| Top price driver (SHAP) | `brand_Vinhomes` |
| Market segments (K-means) | 3 |
| Listings analyzed | 4,362 |
