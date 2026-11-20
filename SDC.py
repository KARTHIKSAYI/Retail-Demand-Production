#!/usr/bin/env python
# coding: utf-8

# In[7]:


# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
print("Libraries imported successfully!")
# 2. LOAD DATASET
df = pd.read_parquet("train.parquet")
print("Dataset loaded successfully!")
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
display(df.head())
print("\nColumn Names:")
print(df.columns.tolist())
print("\nDataset Information:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())



# 3. DATA CLEANING
# Converting date column
df["dt"] = pd.to_datetime(
    df["dt"],
    errors="coerce"
)
# Converting target column to numeric

df["sale_amount"] = pd.to_numeric(
    df["sale_amount"],
    errors="coerce"
)

# Remove rows with missing target or date

df.dropna(
    subset=["dt", "sale_amount"],
    inplace=True
)

# Remove duplicate records using product-store-date keys.
# This assumes one record per product, store and date.

df.drop_duplicates(
    subset=["dt", "store_id", "product_id"],
    inplace=True
)

# Sort records chronologically within each product-store group

df.sort_values(
    by=["store_id", "product_id", "dt"],
    inplace=True
)

df.reset_index(drop=True, inplace=True)

print("\nShape After Cleaning:")
print(df.shape)



# 4. FEATURE ENGINEERING
# Extract date information

df["day_of_week"] = df["dt"].dt.dayofweek
df["month"] = df["dt"].dt.month
df["day_of_month"] = df["dt"].dt.day


# Previous day's sales
df["sales_lag_1"] = (df.groupby(["store_id", "product_id"])["sale_amount"].shift(1))


# Sales from 7 days earlier
df["sales_lag_7"] = (df.groupby(["store_id", "product_id"])["sale_amount"].shift(7))

# Average sales of previous 7 days

df["sales_roll_mean_7"] = (
    df.groupby(["store_id", "product_id"])["sale_amount"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)


# Previous day's stockout information
if "stock_hour6_22_cnt" in df.columns:

    df["stockout_hours_lag_1"] = (
        df.groupby(["store_id", "product_id"])[
            "stock_hour6_22_cnt"
        ].shift(1)
    )


# 5. SELECT INPUT FEATURES
# Columns that must not be used as current-day inputs

excluded_columns = [
    "dt",
    "sale_amount",
    "hours_sale",
    "hours_stock_status",
    "stock_hour6_22_cnt"
]

# Select only numeric columns.
# This automatically excludes array/list columns.

numeric_columns = df.select_dtypes(  include=["number", "bool"]).columns.tolist()

feature_columns = [
    col for col in numeric_columns
    if col not in excluded_columns
]

# Ensure historical sales features are included
print("\nSelected Features:")
print(feature_columns)
# Convert feature columns to numeric

for col in feature_columns:

    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without sufficient sales history
df.dropna(
    subset=[
        "sales_lag_1",
        "sales_lag_7",
        "sales_roll_mean_7"
    ],
    inplace=True
)

df.reset_index(drop=True, inplace=True)


# 6. CREATE X AND Y

X = df[feature_columns].copy()
y = df["sale_amount"].copy()
print("\nInput Data Shape:", X.shape)
print("Target Data Shape:", y.shape)


# 7. TRAIN-TEST SPLIT
# Chronological split: earlier dates for training,
# later dates for testing.

unique_dates = np.sort(df["dt"].unique())

if len(unique_dates) < 2:
    raise ValueError("Not enough unique dates for splitting.")

split_position = int(len(unique_dates) * 0.80)
split_position = max( 1, min(split_position, len(unique_dates) - 1))

split_date = unique_dates[split_position]
train_mask = df["dt"] < split_date
test_mask = df["dt"] >= split_date
X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()
y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()


# Limit training rows to control memory usage
MAX_TRAIN_ROWS = 1000000

if len(X_train) > MAX_TRAIN_ROWS:
    X_train = X_train.iloc[-MAX_TRAIN_ROWS:]
    y_train = y_train.loc[X_train.index]


print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
print("Training starts before:", split_date)
print("\nPreprocessing Completed Successfully!")


# In[26]:


# 1.  MODELS
linear_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())
])


random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])



# 2. TRAIN LINEAR REGRESSION
print("Training Linear Regression...")
linear_model.fit(X_train, y_train)

print("Linear Regression Training Completed!")


# 3. TRAIN RANDOM FOREST


print("\nTraining Random Forest...")
random_forest_model.fit(X_train,y_train)
print("Random Forest Training Completed!")



# 4. STORE MODELS
models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model
}
print("\nBoth Models Trained Successfully!")


# In[27]:


# =====================================================
# 1. CREATE RESULT STORAGE
# =====================================================

results = []

predictions = {}


# =====================================================
# 2. EVALUATE BOTH MODELS
# =====================================================

for name, model in models.items():

    print("\n================================")

    print("MODEL:", name)

    print("================================")


    # Training predictions

    train_pred = model.predict(X_train)


    # Testing predictions

    test_pred = model.predict(X_test)


    # Store testing predictions for plots

    predictions[name] = test_pred


    # ---------------------------------
    # TRAINING METRICS
    # ---------------------------------

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )


    # ---------------------------------
    # TESTING METRICS
    # ---------------------------------

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )


    # ---------------------------------
    # DISPLAY RESULTS
    # ---------------------------------

    print("\nTRAINING RESULTS")

    print("R2 Score:", round(train_r2, 4))

    print("MAE:", round(train_mae, 4))

    print("RMSE:", round(train_rmse, 4))


    print("\nTESTING RESULTS")

    print("R2 Score:", round(test_r2, 4))

    print("MAE:", round(test_mae, 4))

    print("RMSE:", round(test_rmse, 4))


    # Store results

    results.append({

        "Model": name,

        "Train R2": train_r2,

        "Test R2": test_r2,

        "Train MAE": train_mae,

        "Test MAE": test_mae,

        "Train RMSE": train_rmse,

        "Test RMSE": test_rmse

    })


# =====================================================
# 3. COMPARISON TABLE
# =====================================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

display(results_df)


# =====================================================
# 4. SELECT BEST MODEL
# =====================================================

best_model_name = results_df.sort_values(

    by=["Test R2", "Test RMSE"],

    ascending=[False, True]

).iloc[0]["Model"]


best_model = models[best_model_name]


print("\nBest Model:", best_model_name)


# =====================================================
# 5. SAVE MODELS
# =====================================================

os.makedirs("outputs", exist_ok=True)

joblib.dump(
    linear_model,
    "outputs/linear_regression.pkl"
)

joblib.dump(
    random_forest_model,
    "outputs/random_forest.pkl"
)

joblib.dump(
    best_model,
    "outputs/best_demand_model.pkl"
)

joblib.dump(
    feature_columns,
    "outputs/feature_columns.pkl"
)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nModels and results saved successfully!")


# In[28]:


# Run this only if pyarrow is not installed
# !pip install pyarrow
import pandas as pd
import numpy as np
import joblib
import os

# Load evaluation dataset
eval_df = pd.read_parquet("eval.parquet")
print("Evaluation dataset shape:", eval_df.shape)
print("\nColumn names:")
print(eval_df.columns.tolist())
print("\nFirst five rows:")
display(eval_df.head())
print("\nDataset information:")
eval_df.info()


# In[29]:


import pandas as pd
import numpy as np
import joblib
import os

linear_model = joblib.load("outputs/linear_regression.pkl")
random_forest_model = joblib.load("outputs/random_forest.pkl")
best_model = joblib.load("outputs/best_demand_model.pkl")
feature_columns = joblib.load("outputs/feature_columns.pkl")
print("All models loaded successfully!")
print("\nFeatures used during training:")
print(feature_columns)


# In[30]:


missing_features = [
    col for col in feature_columns
    if col not in eval_df.columns
]

print("Missing features:", missing_features)
if len(missing_features) == 0:
    print("All training features are available!")
else:
    print("Some training features need to be created.")


# In[31]:


# Convert date column into datetime format
eval_df["dt"] = pd.to_datetime(eval_df["dt"])

# Extract date-related features
eval_df["day_of_week"] = eval_df["dt"].dt.dayofweek
eval_df["month"] = eval_df["dt"].dt.month
eval_df["day_of_month"] = eval_df["dt"].dt.day

print("Date features created successfully!")

display(eval_df[["dt", "day_of_week", "month", "day_of_month"]].head())


# In[32]:


train_df = pd.read_parquet("train.parquet")
train_df["dt"] = pd.to_datetime(train_df["dt"])

print("Training shape:", train_df.shape)
print("Evaluation shape:", eval_df.shape)

print("\nTraining date range:")
print(train_df["dt"].min(), "to", train_df["dt"].max())

print("\nEvaluation date range:")
print(eval_df["dt"].min(), "to", eval_df["dt"].max())

print("\nTraining columns:")
print(train_df.columns.tolist())


# In[33]:


# Select one store-product combination
sample_store = train_df["store_id"].iloc[0]
sample_product = train_df["product_id"].iloc[0]

sample_data = train_df[
    (train_df["store_id"] == sample_store) &
    (train_df["product_id"] == sample_product)
].sort_values("dt")

display(
    sample_data[
        ["dt", "store_id", "product_id",
         "sale_amount", "stock_hour6_22_cnt"]
    ].head(15)
)


# In[34]:


# Keep a copy of evaluation data
eval_df = eval_df.copy()

# Add a column to identify evaluation rows
train_df["is_eval"] = 0
eval_df["is_eval"] = 1

# Combine both datasets
combined_df = pd.concat(
    [train_df, eval_df],
    ignore_index=True
)

# Sort by store, product and date
combined_df = combined_df.sort_values(
    ["store_id", "product_id", "dt"]
).reset_index(drop=True)

print("Combined dataset shape:", combined_df.shape)


# In[35]:


# Group data by store and product
grouped = combined_df.groupby(["store_id", "product_id"])

# Previous day's sales
combined_df["sales_lag_1"] = grouped["sale_amount"].shift(1)

# Sales from 7 previous rows
combined_df["sales_lag_7"] = grouped["sale_amount"].shift(7)

# Average sales of the previous 7 rows
combined_df["sales_roll_mean_7"] = grouped["sale_amount"].transform(
    lambda x: x.shift(1).rolling(
        window=7,
        min_periods=1
    ).mean()
)

# Previous day's stockout count
combined_df["stockout_hours_lag_1"] = grouped[
    "stock_hour6_22_cnt"
].shift(1)

print("Historical features created successfully!")


# In[36]:


# Select only evaluation rows
eval_features = combined_df[
    combined_df["is_eval"] == 1
].copy()

# Sort back into date and product order
eval_features = eval_features.sort_values(
    ["store_id", "product_id", "dt"]
).reset_index(drop=True)

print("Evaluation shape:", eval_features.shape)

print("\nEvaluation features:")
display(
    eval_features[
        [
            "dt",
            "store_id",
            "product_id",
            "sale_amount",
            "sales_lag_1",
            "sales_lag_7",
            "sales_roll_mean_7",
            "stockout_hours_lag_1"
        ]
    ].head(10)
)


# In[37]:


print("Missing values in training features:")
print( eval_features[feature_columns].isnull().sum())

print("\nTotal missing values:")
print(eval_features[feature_columns].isnull().sum().sum())


# In[38]:


# Select exactly the features used during training
X_eval = eval_features[feature_columns].copy()
# Convert feature columns to numeric format
for col in feature_columns:
    X_eval[col] = pd.to_numeric(
        X_eval[col],
        errors="coerce"
    )

print("Evaluation features prepared!")
print("X_eval shape:", X_eval.shape)
display(X_eval.head())


# In[39]:


# Linear Regression predictions
linear_predictions = linear_model.predict(X_eval)
# Random Forest predictions
rf_predictions = random_forest_model.predict(X_eval)
# Best model predictions
best_predictions = best_model.predict(X_eval)

print("Predictions generated successfully!")
print("\nFirst 10 predictions:")
print("Linear Regression:", linear_predictions[:10])
print("Random Forest:", rf_predictions[:10])
print("Best Model:", best_predictions[:10])


# In[40]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Actual values
y_actual = eval_features["sale_amount"]

# Calculate metrics
models = {
    "Linear Regression": linear_predictions,
    "Random Forest": rf_predictions,
    "Best Model": best_predictions
}

evaluation_results = []
for model_name, predictions in models.items():
    r2 = r2_score(y_actual, predictions)
    mae = mean_absolute_error(y_actual, predictions)
    rmse = np.sqrt( mean_squared_error(y_actual, predictions))

    evaluation_results.append({
        "Model": model_name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

evaluation_df = pd.DataFrame(evaluation_results)
display(evaluation_df)


# In[42]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Predictions on training data
linear_train_pred = linear_model.predict(X_train)
rf_train_pred = random_forest_model.predict(X_train)

# Predictions on testing data
linear_test_pred = linear_model.predict(X_test)
rf_test_pred = random_forest_model.predict(X_test)


# Function to calculate performance
def calculate_metrics(actual, predicted):

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return r2, mae, rmse


# Calculate results
results = []
models = {
    "Linear Regression": (linear_train_pred, linear_test_pred),
    "Random Forest": (rf_train_pred, rf_test_pred)
}

for name, (train_pred, test_pred) in models.items():
    train_r2, train_mae, train_rmse = calculate_metrics(y_train, train_pred)

    test_r2, test_mae, test_rmse = calculate_metrics(y_test, test_pred)

    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })


results_df = pd.DataFrame(results)
display(results_df.round(4))


# In[43]:


import matplotlib.pyplot as plt

x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar( x - width/2, results_df["Train R2"], width, label="Training R2")

plt.bar(x + width/2, results_df["Test R2"], width, label="Testing R2")

plt.xticks(x, results_df["Model"])

plt.ylabel("R2 Score")
plt.title("Training vs Testing Performance")

plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.show()


# In[44]:


# =====================================================
# 1. SALES DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 5))

plt.hist( df["sale_amount"], bins=40)

plt.title("Sales Distribution")

plt.xlabel("Sale Amount")

plt.ylabel("Frequency")

plt.show()


# =====================================================
# 2. DAILY SALES TREND
# =====================================================

daily_sales = df.groupby("dt")["sale_amount"].mean()

plt.figure(figsize=(12, 5))

daily_sales.plot()

plt.title("Daily Average Sales Trend")

plt.xlabel("Date")

plt.ylabel("Average Sale Amount")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =====================================================
# 3. ACTUAL VS PREDICTED - LINEAR REGRESSION
# =====================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions["Linear Regression"],
    alpha=0.3
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Linear Regression: Actual vs Predicted")

plt.show()


# =====================================================
# 4. ACTUAL VS PREDICTED - RANDOM FOREST
# =====================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    predictions["Random Forest"],
    alpha=0.3
)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Random Forest: Actual vs Predicted")

plt.show()


# =====================================================
# 5. TRAIN R2 VS TEST R2
# =====================================================

comparison = results_df.set_index("Model")

comparison[["Train R2", "Test R2"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Training R2 vs Testing R2")

plt.ylabel("R2 Score")

plt.xticks(rotation=0)

plt.show()


# =====================================================
# 6. MAE COMPARISON
# =====================================================

comparison[["Train MAE", "Test MAE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("MAE Comparison")

plt.ylabel("Mean Absolute Error")

plt.xticks(rotation=0)

plt.show()


# =====================================================
# 7. RMSE COMPARISON
# =====================================================

comparison[["Train RMSE", "Test RMSE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("RMSE Comparison")

plt.ylabel("Root Mean Squared Error")

plt.xticks(rotation=0)

plt.show()


print("All Visualizations Completed!")


# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate 100 random samples
random_data = pd.DataFrame(
    np.random.uniform(0, 10, (100, len(feature_columns))),
    columns=feature_columns
)

# Create nonlinear patterns in selected features
random_data["discount"] = np.random.uniform(0, 1, 100)
random_data["precpt"] = np.random.uniform(0, 20, 100)

print("Random nonlinear input data created!")

display(random_data.head())


# In[46]:


# Predict using already-trained models
linear_predictions = linear_model.predict(random_data)

rf_predictions = random_forest_model.predict(random_data)

best_predictions = best_model.predict(random_data)

# Store predictions
prediction_results = pd.DataFrame({
    "Linear_Regression": linear_predictions,
    "Random_Forest": rf_predictions,
    "Best_Model": best_predictions
})

display(prediction_results.head(10))


# In[47]:


plt.figure(figsize=(10, 5))

plt.plot(
    prediction_results["Linear_Regression"],
    label="Linear Regression"
)

plt.plot(
    prediction_results["Random_Forest"],
    label="Random Forest"
)

plt.plot(
    prediction_results["Best_Model"],
    label="Best Model"
)

plt.xlabel("Random Sample")
plt.ylabel("Predicted Sale Amount")

plt.title("Predictions on New Random Data")

plt.legend()
plt.grid()

plt.show()


# In[48]:


import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Load trained models
linear_model = joblib.load("outputs/linear_regression.pkl")
random_forest_model = joblib.load("outputs/random_forest.pkl")

# Load feature columns
feature_columns = joblib.load("outputs/feature_columns.pkl")

print("Models loaded successfully!")


# In[49]:


models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model
}

results = []

for name, model in models.items():

    # Training predictions
    train_predictions = model.predict(X_train)

    # Testing predictions
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train MAE": train_mae,
        "Test MAE": test_mae
    })

results_df = pd.DataFrame(results)

print(results_df)


# In[50]:


best_model_name = results_df.sort_values(
    by=["Test R2", "Test MAE"],
    ascending=[False, True]
).iloc[0]["Model"]

best_model = models[best_model_name]

print("Best Model:", best_model_name)


# In[51]:


joblib.dump(best_model, "outputs/best_demand_model.pkl")

print("Best model saved successfully!")


# In[52]:


selected_model = joblib.load("outputs/best_demand_model.pkl")

print("Selected model loaded successfully!")
print("Model type:", type(selected_model).__name__)


# In[ ]:





# In[ ]:





# In[ ]:




