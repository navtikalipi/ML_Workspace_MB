# ============================================
# Loan Approval FastAPI with SQLite Logging
# ============================================

import joblib
import pandas as pd
import sqlite3
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------
# Load trained model (pipeline)
# --------------------------------------------
model_pipeline = joblib.load("loan_model.pkl")

# --------------------------------------------
# Initialize FastAPI app
# --------------------------------------------
app = FastAPI(
    title="Loan Approval API",
    description="Predict loan approval and store results in SQLite",
    version="1.0"
)

# --------------------------------------------
# Initialize SQLite database
# --------------------------------------------
DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_term INTEGER,
        credit_history INTEGER,
        married TEXT,
        self_employed TEXT,
        education TEXT,
        property_area TEXT,
        loan_status TEXT,
        approval_probability REAL,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

# Run DB initialization at startup
init_db()

# --------------------------------------------
# Request schema (input validation)
# --------------------------------------------
class LoanApplication(BaseModel):
    ApplicantIncome: float = Field(..., ge=0)
    CoapplicantIncome: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=0)
    Loan_Amount_Term: int = Field(..., gt=0)
    Credit_History: int = Field(..., ge=0, le=1)
    Married: str
    Self_Employed: str
    Education: str
    Property_Area: str

# --------------------------------------------
# Helper: Save prediction to DB
# --------------------------------------------
def save_prediction(data: LoanApplication, status: str, probability: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO loan_predictions (
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        married,
        self_employed,
        education,
        property_area,
        loan_status,
        approval_probability,
        created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        data.Married,
        data.Self_Employed,
        data.Education,
        data.Property_Area,
        status,
        probability,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

# --------------------------------------------
# Prediction endpoint
# --------------------------------------------
@app.post("/predict")
def predict_loan(data: LoanApplication):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Predict probability
        probability = model_pipeline.predict_proba(df)[0][1]

        # Business decision threshold
        THRESHOLD = 0.6
        loan_status = "Approved" if probability >= THRESHOLD else "Rejected"

        # Save to database
        save_prediction(data, loan_status, probability)

        return {
            "loan_status": loan_status,
            "approval_probability": round(float(probability), 4)
        }

    except Exception as e:
        # Do NOT hide the error (important for debugging)
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------
# Health check endpoint (optional but useful)
# --------------------------------------------
@app.get("/health")
def health():
    return {"status": "OK"}
