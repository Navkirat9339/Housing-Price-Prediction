# California Housing Price Prediction

A machine learning project that predicts median house values for California census block groups using regression models, with a Streamlit web application for interactive predictions.

## Project Overview

This project explores the complete machine learning workflow, from exploratory data analysis and implementing linear regression from scratch to training, tuning, and evaluating tree-based ensemble models.

The final application uses a tuned XGBoost regression model to estimate median house values from eight housing and geographic features.

## Dataset

The project uses the California Housing dataset available through Scikit-learn.

- **Observations:** 20,640
- **Input features:** 8
- **Target:** Median house value (`MedHouseVal`)
- **Data source:** Scikit-learn's California Housing dataset

The target is expressed in units of $100,000. The dataset represents census block groups rather than individual properties, and its target values are capped near $500,000.

## Models Explored

1. Linear Regression implemented from scratch
2. Linear Regression with feature scaling
3. Decision Tree Regression
4. Random Forest Regression
5. Gradient Boosting Regression
6. XGBoost Regression

The project includes exploratory data analysis, feature analysis, model evaluation, hyperparameter tuning, and error analysis.

## Final Model Performance

The tuned XGBoost model achieved the following results on the held-out test split:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.2833 |
| Root Mean Squared Error (RMSE) | 0.4342 |
| R² Score | 0.8561 |

Because the target is measured in units of $100,000, the MAE corresponds to approximately **$28,330**.

These results describe performance on this dataset and split; they do not guarantee accuracy for real-world property valuations.

## Project Structure

```text
Housing-Price-Prediction/
├── app/
│   └── streamlit_app.py
├── models/
│   └── xgboost_housing_model.joblib
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_linear_regression_from_scratch.ipynb
│   ├── 03_feature_scaling_and_regression.ipynb
│   ├── 04_decision_tree_regression.ipynb
│   ├── 05_random_forest_regressor.ipynb
│   ├── 06_gradient_boosting_regression.ipynb
│   └── 07_xgboost_regression.ipynb
├── src/
│   └── predict.py
├── test_predict.py
├── requirements.txt
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Housing-Price-Prediction
```

Replace the placeholder with your repository's actual URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run a prediction test

```bash
python test_predict.py
```

### 5. Launch the Streamlit application

```bash
python -m streamlit run app/streamlit_app.py
```

The application allows users to enter housing and geographic features and obtain a predicted median house value.

## Key Learnings

- Exploratory data analysis and feature relationships
- Implementing linear regression and understanding gradient descent
- Comparing linear and tree-based regression models
- Hyperparameter tuning and evaluation using MAE, RMSE, and R²
- Feature importance and residual analysis
- Saving trained models and integrating them into a Streamlit application

## Limitations

- The dataset's target values are capped near $500,000, limiting the model's ability to distinguish higher-valued areas.
- Predictions are based on historical census block group data, not individual property details.
- The model does not account for current market conditions, property-specific attributes, or all factors affecting house prices.
- Predictions should be treated as estimates for educational purposes, not professional property valuations.

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost, Joblib, Jupyter Notebook, and Streamlit.
