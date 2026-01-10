import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
Welcome! This tool predicts whether a credit card transaction is **fraudulent** or not.
Fill in the details below. Default values are provided for guidance.
""")

st.divider()
st.subheader("🧾 Transaction Details")

# Example friendly inputs
merchant = st.selectbox("Merchant", [100, 120, 140, 160], help="Select the merchant ID. (Auto-encoded)")
category = st.selectbox("Transaction Category", [1, 2, 3, 4], help="Select the type of purchase.")
amt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
gender = st.selectbox("Gender of Card Holder", ["Female", "Male"], help="Select the gender.")
lat = st.number_input("Customer Latitude", value=40.0)
long = st.number_input("Customer Longitude", value=-73.0)
city_pop = st.number_input("City Population", value=50000)
job = st.selectbox("Job Type", [100, 150, 200, 220], help="Select your job category.")
merch_lat = st.number_input("Merchant Latitude", value=40.5)
merch_long = st.number_input("Merchant Longitude", value=-73.5)
trans_hour = st.slider("Transaction Hour", 0, 23, 12)
trans_day = st.slider("Transaction Day", 1, 31, 15)
trans_month = st.slider("Transaction Month", 1, 12, 6)

# Map user-friendly inputs to encoded values if necessary
gender_val = 1 if gender == "Male" else 0

# Predict button
if st.button("🔍 Check Transaction"):
    input_data = pd.DataFrame([[merchant, category, amt, gender_val, lat, long,
                                city_pop, job, merch_lat, merch_long,
                                trans_hour, trans_day, trans_month]],
        columns=[
            'merchant','category','amt','gender','lat','long',
            'city_pop','job','merch_lat','merch_long',
            'trans_hour','trans_day','trans_month'
        ])
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # probability of fraud

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected!\nConfidence: {prob*100:.2f}%")
    else:
        st.success(f"✅ Transaction is Legitimate\nConfidence: {(1-prob)*100:.2f}%")

st.divider()
st.caption("Built with ❤️ using Streamlit and Machine Learning. Designed for first-time users.")
