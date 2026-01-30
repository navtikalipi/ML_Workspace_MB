import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DB_NAME = "house_predictions.db"

PRICE_MODEL = joblib.load("models/price_model.pkl")
QUICKSALE_MODEL = joblib.load("models/quicksale_model.pkl")

app = FastAPI(title="House Sales API", version="1.0")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS price_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload_json TEXT,
        predicted_price REAL,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS quicksale_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload_json TEXT,
        predicted_label INTEGER,
        predicted_probability REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


init_db()


# -------- Request Schemas --------
class HouseBase(BaseModel):
    Square_Footage: float = Field(..., ge=0)
    Bedrooms: int = Field(..., ge=0)
    Bathrooms: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)
    Garage_Spaces: int = Field(..., ge=0)
    Lot_Size: float = Field(..., ge=0)
    Floors: int = Field(..., ge=0)
    Neighborhood_Rating: float = Field(..., ge=0)
    Condition: float = Field(..., ge=0)
    School_Rating: float = Field(..., ge=0)
    Has_Pool: str
    Renovated: str
    Location_Type: str
    Distance_To_Center_KM: float = Field(..., ge=0)


class HouseForQuickSale(HouseBase):
    Price: float = Field(..., ge=0)


# -------- DB helpers --------
def insert_price(payload: dict, predicted_price: float):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO price_predictions(payload_json, predicted_price, created_at) VALUES (?, ?, ?)",
        (str(payload), float(predicted_price), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def insert_quicksale(payload: dict, label: int, prob: float):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO quicksale_predictions(payload_json, predicted_label, predicted_probability, created_at) VALUES (?, ?, ?, ?)",
        (str(payload), int(label), float(prob), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


@app.get("/health")
def health():
    return {"status": "OK"}


# -------- Predictions --------
@app.post("/predict/price")
def predict_price(data: HouseBase):
    try:
        df = pd.DataFrame([data.dict()])
        pred = PRICE_MODEL.predict(df)[0]

        insert_price(data.dict(), pred)

        return {"predicted_price": round(float(pred), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/quicksale")
def predict_quicksale(data: HouseForQuickSale):
    try:
        df = pd.DataFrame([data.dict()])
        prob = QUICKSALE_MODEL.predict_proba(df)[0][1]
        label = 1 if prob >= 0.5 else 0

        insert_quicksale(data.dict(), label, prob)

        return {
            "sold_within_week": "Yes" if label == 1 else "No",
            "probability": round(float(prob), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------- History --------
@app.get("/history/price")
def history_price():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, payload_json, predicted_price, created_at FROM price_predictions ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
    return {"rows": rows}


@app.get("/history/quicksale")
def history_quicksale():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, payload_json, predicted_label, predicted_probability, created_at FROM quicksale_predictions ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
    return {"rows": rows}
