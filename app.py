import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from scipy.sparse import hstack

app = FastAPI()

# ===== LOAD =====
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf.pkl")
columns = joblib.load("columns.pkl")

# ===== PREPROCESS =====
def preprocess(data: dict):
    df = pd.DataFrame([data])

    # ===== TEXT =====
    df["title"] = df.get("title", "").fillna("")
    df["description"] = df.get("description", "").fillna("")
    df["text"] = df["title"] + " " + df["description"]

    # ===== CITY NORMALIZE =====
    df["city"] = df.get("city", "hcm")
    df["city"] = df["city"].apply(
        lambda x: "hcm" if "hcm" in str(x).lower() else (
            "hanoi" if "ha noi" in str(x).lower() else x
        )
    )

    # ===== NUMERIC FEATURES =====
    num_cols = [
        'area_m2', 'bedrooms', 'bathrooms',
        'dist_cbd_km', 'dist_metro_km', 'dist_river_lake_km',
        'dist_highway_km', 'dist_airport_km',
        'amenity_score', 'description_length',
        'total_apartment', 'construction_density'
    ]

    for col in num_cols:
        df[col] = df.get(col, 0)

    # ===== BINARY FEATURES =====
    binary_cols = [
        'Metro_city','luxury','river_view','investment',
        'modern_urban','full_furniture','high_floor'
    ]

    for col in binary_cols:
        df[col] = df.get(col, 0)

    # ===== LOG TRANSFORM =====
    df['total_apartment'] = np.log1p(df['total_apartment'])
    df['description_length'] = np.log1p(df['description_length'])

    # ===== TEXT VECTOR =====
    text_data = df["text"].copy()

    # ===== ONE HOT =====
    df = pd.get_dummies(df)

    # ===== ALIGN COLUMN =====
    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]

    # ===== SCALE =====
    df[num_cols] = df[num_cols].astype(float)
    df[num_cols] = scaler.transform(df[num_cols])

    # ===== TO NUMPY =====
    df_np = df.astype(float).to_numpy()

    # ===== TFIDF =====
    text_vec = tfidf.transform(text_data)

    # ===== COMBINE =====
    final = hstack([df_np, text_vec])

    return final


# ===== API =====
@app.post("/predict")
def predict(data: dict):
    try:
        record_id = data.get("id")

        X = preprocess(data)
        pred_log = model.predict(X)[0]
        pred_price = np.exp(pred_log)

        return {
            "id": record_id,
            "predicted_price": float(pred_price)
        }

    except Exception as e:
        return {
            "id": data.get("id"),
            "error": str(e)
        }