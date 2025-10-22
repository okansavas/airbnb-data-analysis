# run_full_project.py (Streamlit Cloud uyumlu versiyon)
import os
import streamlit as st
import pickle
import pandas as pd

# ========== CONFIG ==========
MODEL_PATH = "models/airbnb_model.pkl"
DATA_PATH = "data/AB_NYC_2019.csv"

# ========== HELPER FUNCTIONS ==========
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    else:
        st.error("❌ Model file not found! Please check models/airbnb_model.pkl")
        return None

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error("❌ Dataset not found! Please check data/AB_NYC_2019.csv")
        return None

# ========== STREAMLIT UI ==========
st.title("🏠 Airbnb Price Prediction Dashboard")
st.markdown("Explore and predict Airbnb listing prices in NYC")

data = load_data()
model = load_model()

if data is not None:
    st.write("### Sample of Dataset:")
    st.dataframe(data.head())

    st.sidebar.header("Make a Prediction")
    room_type = st.sidebar.selectbox("Room Type", data["room_type"].unique())
    neighbourhood = st.sidebar.selectbox("Neighbourhood Group", data["neighbourhood_group"].unique())
    minimum_nights = st.sidebar.number_input("Minimum Nights", min_value=1, value=3)
    number_of_reviews = st.sidebar.number_input("Number of Reviews", min_value=0, value=10)

    if st.sidebar.button("Predict"):
        if model is not None:
            # Dummy example — replace with your feature pipeline
            sample = pd.DataFrame({
                "room_type": [room_type],
                "neighbourhood_group": [neighbourhood],
                "minimum_nights": [minimum_nights],
                "number_of_reviews": [number_of_reviews]
            })
            try:
                price = model.predict(sample)[0]
                st.success(f"💰 Predicted Price: ${price:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Model not loaded.")
