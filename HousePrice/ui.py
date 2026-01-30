import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="House Sales Predictor", page_icon="🏠", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🏠 House Sales Prediction App</h1>",
    unsafe_allow_html=True
)
st.caption("Frontend: Streamlit • Backend: FastAPI • DB: SQLite")


def house_form(include_price: bool):
    col1, col2, col3 = st.columns(3)

    with col1:
        Square_Footage = st.number_input("Square Footage", min_value=0.0, step=50.0, value=2000.0)
        Bedrooms = st.number_input("Bedrooms", min_value=0, step=1, value=3)
        Bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5, value=2.0)
        Age = st.number_input("Age (years)", min_value=0.0, step=1.0, value=10.0)

    with col2:
        Garage_Spaces = st.number_input("Garage Spaces", min_value=0, step=1, value=1)
        Lot_Size = st.number_input("Lot Size", min_value=0.0, step=0.1, value=1.0)
        Floors = st.number_input("Floors", min_value=0, step=1, value=1)
        Distance_To_Center_KM = st.number_input("Distance to Center (KM)", min_value=0.0, step=0.5, value=10.0)

    with col3:
        Neighborhood_Rating = st.number_input("Neighborhood Rating", min_value=0.0, step=0.1, value=7.0)
        Condition = st.number_input("Condition", min_value=0.0, step=0.1, value=7.0)
        School_Rating = st.number_input("School Rating", min_value=0.0, step=0.1, value=7.0)
        Location_Type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
        Has_Pool = st.selectbox("Has Pool", ["Yes", "No"])
        Renovated = st.selectbox("Renovated", ["Yes", "No"])

    payload = {
        "Square_Footage": Square_Footage,
        "Bedrooms": int(Bedrooms),
        "Bathrooms": Bathrooms,
        "Age": Age,
        "Garage_Spaces": int(Garage_Spaces),
        "Lot_Size": Lot_Size,
        "Floors": int(Floors),
        "Neighborhood_Rating": Neighborhood_Rating,
        "Condition": Condition,
        "School_Rating": School_Rating,
        "Has_Pool": Has_Pool,
        "Renovated": Renovated,
        "Location_Type": Location_Type,
        "Distance_To_Center_KM": Distance_To_Center_KM,
    }

    if include_price:
        payload["Price"] = st.number_input("Price (needed for quick sale model)", min_value=0.0, step=1000.0, value=750000.0)

    return payload


tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "⚡ Quick Sale Prediction", "📜 History"])

# -------- Tab 1: Price Prediction --------
with tab1:
    st.subheader("Predict House Price (Linear Regression)")
    with st.form("price_form"):
        payload = house_form(include_price=False)
        submitted = st.form_submit_button("Predict Price")

    if submitted:
        with st.spinner("Calling API..."):
            try:
                r = requests.post(f"{API_BASE}/predict/price", json=payload, timeout=10)
                if r.status_code == 200:
                    out = r.json()
                    st.success(f"Predicted Price: ${out['predicted_price']:,.2f}")
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"API error: {e}")

# -------- Tab 2: Quick Sale --------
with tab2:
    st.subheader("Predict if House Sells Within 1 Week (Logistic Regression)")
    with st.form("qs_form"):
        payload = house_form(include_price=True)
        submitted = st.form_submit_button("Predict Quick Sale")

    if submitted:
        with st.spinner("Calling API..."):
            try:
                r = requests.post(f"{API_BASE}/predict/quicksale", json=payload, timeout=10)
                if r.status_code == 200:
                    out = r.json()
                    sold = out["sold_within_week"]
                    prob = out["probability"]

                    if sold == "Yes":
                        st.success(f"Likely to sell within 1 week ✅  (Probability: {prob:.2%})")
                    else:
                        st.warning(f"Not likely to sell within 1 week ⚠️ (Probability: {prob:.2%})")

                    st.progress(min(max(prob, 0.0), 1.0))
                else:
                    st.error(r.text)
            except Exception as e:
                st.error(f"API error: {e}")

# -------- Tab 3: History --------
with tab3:
    st.subheader("Recent Predictions (from SQLite via API)")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 💰 Price Predictions")
        try:
            r = requests.get(f"{API_BASE}/history/price", timeout=10)
            rows = r.json()["rows"]
            df = pd.DataFrame(rows, columns=["id", "payload", "predicted_price", "created_at"])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"History error: {e}")

    with colB:
        st.markdown("### ⚡ Quick Sale Predictions")
        try:
            r = requests.get(f"{API_BASE}/history/quicksale", timeout=10)
            rows = r.json()["rows"]
            df = pd.DataFrame(rows, columns=["id", "payload", "label", "probability", "created_at"])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"History error: {e}")
