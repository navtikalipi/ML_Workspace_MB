from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI(title="Customer Clustering API")

kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


cluster_to_segment = {
    0: "High-Value Loyal",
    1: "Value-Seeking Regular",
    2: "Price-Sensitive Occasional"
}

cluster_to_offer = {
    0: "Exclusive early access to new products + Premium membership with free express delivery",
    1: "Festival discounts (10–15%) + Loyalty reward points on every purchase",
    2: "Flash sales and coupon-based discounts + Free shipping on minimum order value"
}


@app.post("/predict")
def predict_customer(customer: dict):
    """
    Example input:
    {
        "Age": 35,
        "Gender": 1,
        "City": 2,
        "AnnualIncome": 900000,
        "TotalSpent": 220000,
        "MonthlyPurchases": 6,
        "AvgOrderValue": 4500,
        "AppTimeMinutes": 60,
        "DiscountUsage": 1,
        "PreferredShoppingTime": 1
    }
    """

    df_new = pd.DataFrame([customer])

    # Apply same scaling as training
    X_scaled = scaler.transform(df_new)

    # Predict cluster
    cluster = int(kmeans.predict(X_scaled)[0])

    # Map to segment & offer
    segment = cluster_to_segment[cluster]
    offer = cluster_to_offer[cluster]

    return {
        "Predicted_Cluster": cluster,
        "Customer_Segment": segment,
        "Suggested_Offer": offer
    }
