# =====================================
# 1. IMPORT REQUIRED LIBRARIES
# =====================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# =====================================
# 2. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Retail Demand Prediction API",
    description="Predict product demand using Machine Learning",
    version="1.0"
)


# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "best_demand_model.pkl")

print("Trained model loaded successfully!")


# =====================================
# 4. DEFINE INPUT FEATURES
# =====================================

class ProductInput(BaseModel):

    city_id: int
    store_id: int
    management_group_id: int

    first_category_id: int
    second_category_id: int
    third_category_id: int
    product_id: int

    discount: float
    holiday_flag: int
    activity_flag: int

    precpt: float
    avg_temperature: float
    avg_humidity: float
    avg_wind_level: float

    day_of_week: int
    month: int
    day_of_month: int

    sales_lag_1: float
    sales_lag_7: float
    sales_roll_mean_7: float
    stockout_hours_lag_1: float


# =====================================
# 5. PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict_demand(product: ProductInput):

    # Convert input into dictionary
    product_data = product.model_dump()

    # Convert dictionary into DataFrame
    input_data = pd.DataFrame([product_data])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Return prediction
    return {
        "status": "success",
        "predicted_demand": round(float(prediction), 4)
    }