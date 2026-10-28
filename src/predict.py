from pathlib import Path

import joblib
import pandas as pd

# Locate the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_housing_model.joblib"
)

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

def load_model():
    """Load and return the saved XGBoost model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

def predict_house_value(model, features):
    """Predict the median house value for one observation."""
    input_data = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    prediction = model.predict(input_data)[0]

    # Calfornia Housing targets are in units of $100,000.
    return float(prediction)