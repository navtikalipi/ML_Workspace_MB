import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL & SCALER ----------------
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- CLUSTER & OFFER MAPPING ----------------
cluster_to_segment = {
    0: "High-Value Loyal",
    1: "Value-Seeking Regular",
    2: "Price-Sensitive Occasional"
}

cluster_to_offer = {
    0: "Exclusive early access + Premium membership with free express delivery",
    1: "Festival discounts (10–15%) + Loyalty reward points",
    2: "Flash sales + Coupons + Free shipping on minimum order value"
}

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Customer Clustering App", layout="wide")

st.title("🛍️ Customer Clustering & Personalized Offers")

tabs = st.tabs(["📊 Existing Customers", "🧑 New Customer Prediction"])

# ======================================================
# TAB 1: EXISTING CUSTOMERS
# ======================================================
with tabs[0]:
    st.subheader("Existing Customer Segments")

    df_existing = pd.read_csv("Customer_Offers_Existing_Customers.csv")

    st.dataframe(df_existing)

    st.markdown("### Cluster Distribution")
    st.bar_chart(df_existing["Cluster"].value_counts().sort_index())

# ======================================================
# TAB 2: NEW CUSTOMER PREDICTION
# ======================================================
with tabs[1]:
    st.subheader("Predict Segment for New Customer")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        annual_income = st.number_input("Annual Income", value=500000)
        total_spent = st.number_input("Total Amount Spent", value=100000)
        monthly_purchases = st.number_input("Monthly Purchases", value=3)
        avg_order_value = st.number_input("Average Order Value", value=3000)

    with col2:
        app_time = st.number_input("App Time (minutes/day)", value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        city = st.selectbox("City", ["City_0", "City_1", "City_2", "City_3"])
        discount = st.selectbox("Discount Usage", ["Low", "Medium", "High"])
        shopping_time = st.selectbox("Preferred Shopping Time", ["Day", "Night"])

    # ---------------- ENCODING (same logic as training) ----------------
    gender_map = {"Male": 0, "Female": 1}
    discount_map = {"Low": 0, "Medium": 1, "High": 2}
    shopping_time_map = {"Day": 0, "Night": 1}
    city_map = {"City_0": 0, "City_1": 1, "City_2": 2, "City_3": 3}

    if st.button("Predict Customer Segment"):
        new_customer = pd.DataFrame([{
            "Age": age,
            "Gender": gender_map[gender],
            "City": city_map[city],
            "AnnualIncome": annual_income,
            "TotalSpent": total_spent,
            "MonthlyPurchases": monthly_purchases,
            "AvgOrderValue": avg_order_value,
            "AppTimeMinutes": app_time,
            "DiscountUsage": discount_map[discount],
            "PreferredShoppingTime": shopping_time_map[shopping_time]
        }])

        X_scaled = scaler.transform(new_customer)
        cluster = int(kmeans.predict(X_scaled)[0])

        st.success(f"🎯 Customer Segment: {cluster_to_segment[cluster]}")
        st.info(f"🎁 Suggested Offer: {cluster_to_offer[cluster]}")
