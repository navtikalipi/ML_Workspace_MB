import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report


DATA_PATH = "house_sales.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PRICE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
QUICKSALE_MODEL_PATH = os.path.join(MODEL_DIR, "quicksale_model.pkl")


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    # Common feature groups
    numeric_cols = [
        "Square_Footage", "Bedrooms", "Bathrooms", "Age", "Garage_Spaces",
        "Lot_Size", "Floors", "Neighborhood_Rating", "Condition",
        "School_Rating", "Distance_To_Center_KM"
    ]
    categorical_cols = [
        "Has_Pool", "Renovated", "Location_Type"
    ]

    # -------- Use case 1: Linear Regression (Price) --------
    X_price = df[numeric_cols + categorical_cols]
    y_price = df["Price"]

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_price, y_price, test_size=0.2, random_state=42
    )

    preprocessor_price = build_preprocessor(numeric_cols, categorical_cols)

    price_pipeline = Pipeline([
        ("preprocess", preprocessor_price),
        ("model", LinearRegression())
    ])

    price_pipeline.fit(Xp_train, yp_train)
    yp_pred = price_pipeline.predict(Xp_test)

    print("\n=== Linear Regression: Price ===")
    print("MAE:", mean_absolute_error(yp_test, yp_pred))
    print("R2 :", r2_score(yp_test, yp_pred))

    joblib.dump(price_pipeline, PRICE_MODEL_PATH)
    print(f"✅ Saved: {PRICE_MODEL_PATH}")

    # -------- Use case 2: Logistic Regression (Quick Sale) --------
    # Uses same features + Price
    X_qs = df[numeric_cols + categorical_cols + ["Price"]]
    y_qs = df["Sold_Within_Week"].astype(int)

    Xq_train, Xq_test, yq_train, yq_test = train_test_split(
        X_qs, y_qs, test_size=0.2, random_state=42, stratify=y_qs
    )

    preprocessor_qs = build_preprocessor(numeric_cols + ["Price"], categorical_cols)

    quicksale_pipeline = Pipeline([
        ("preprocess", preprocessor_qs),
        ("model", LogisticRegression(max_iter=1000))
    ])

    quicksale_pipeline.fit(Xq_train, yq_train)
    yq_pred = quicksale_pipeline.predict(Xq_test)

    print("\n=== Logistic Regression: Sold_Within_Week ===")
    print("Accuracy:", accuracy_score(yq_test, yq_pred))
    print(classification_report(yq_test, yq_pred))

    joblib.dump(quicksale_pipeline, QUICKSALE_MODEL_PATH)
    print(f"✅ Saved: {QUICKSALE_MODEL_PATH}")


if __name__ == "__main__":
    main()
