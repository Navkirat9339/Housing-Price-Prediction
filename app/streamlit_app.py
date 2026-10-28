import sys
from pathlib import Path

import streamlit as st

# Allow importing src when Streamlit runs from the app folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_house_value

st.set_page_config(
    page_title="California Housing Value Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("California Housing Value Predictor")

st.write(
    "Estimate the median house value of a California"
    "census block group using a trained XGBoost model."
)

st.info(
    "This model estimates area-level median house values,"
    "not the selling prices of an individual property."
)

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

st.subheader("Enter Housing Features")

med_inc = st.number_input(
    "Median Income (dataset units)",
    min_value = 0.0,
    value = 5.0,
    step = 0.1
)

house_age = st.number_input(
    "Median House Age (years)",
    min_value=0.0,
    value=30.0,
    step=1.0
)

ave_rooms = st.number_input(
    "Average Rooms per Household",
    min_value=0.0,
    value=5.0,
    step=0.1
)

ave_bedrms = st.number_input(
    "Average Bedrooms per Household",
    min_value=0.0,
    value=1.0,
    step=0.1
)

population = st.number_input(
    "Block Group Population",
    min_value=0,
    value=1000,
    step=100
)

ave_occup = st.number_input(
    "Average Occupancy per Household",
    min_value=0.0,
    value=3.0,
    step=0.1
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=34.05,
    step=0.01
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-118.25,
    step=0.01
)

if st.button("Predict House Value", type="primary"):

    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    prediction = predict_house_value(model, features)

    estimated_value = prediction * 100_000

    st.success(
        f"Estimated Median House Value: ${estimated_value:,.0f}"
    )

    st.caption(
        "This is a statistical estimate based on historical "
        "California Housing data. Actual values may differ "
        "substantially, especially for unusual observations"
    )