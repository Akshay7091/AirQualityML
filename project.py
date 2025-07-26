import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

try:
    df = pd.read_csv('data_date.csv')
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())

    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    print("\n'Date' column converted to datetime.")

except FileNotFoundError:
    print("Error: data_date.csv not found. Please make sure the file is in the correct directory.")
    exit()

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
df['DayOfYear'] = df['Date'].dt.dayofyear

print("\nDataFrame after adding date-based features:")
print(df.head())

X = df[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Country']]
y = df['AQI Value']

categorical_features = ['Country']
numerical_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep numerical features as they are
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility of results
    # n_jobs=-1: uses all available CPU cores for faster training
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("\nTraining the Air Quality Prediction model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(r2)

print(f"\n--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print("\nMAE of 15.63 means on average, your predictions are off by about 15-16 AQI points.")
print("R-squared of 0.62 means approximately 62% of the variance in AQI values can be explained by the model.")

results_df = pd.DataFrame({'Actual AQI': y_test, 'Predicted AQI': y_pred})
print("\nFirst 10 Actual vs. Predicted AQI Values from Test Set:")
print(results_df.head(10))

print("\n--- How to make new predictions ---")
print("To predict the AQI for new data, you need to provide the 'Date' and 'Country'.")
print("The model will automatically extract date features and apply the necessary encoding.")

new_data = pd.DataFrame({
    'Date': ['2027-09-29', '2024-01-15'],
    'Country': ['India', 'United States of America']
})

new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data['Year'] = new_data['Date'].dt.year
new_data['Month'] = new_data['Date'].dt.month
new_data['Day'] = new_data['Date'].dt.day
new_data['DayOfWeek'] = new_data['Date'].dt.dayofweek
new_data['DayOfYear'] = new_data['Date'].dt.dayofyear

new_data_features = new_data[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Country']]
predicted_aqi = model_pipeline.predict(new_data_features)

print(f"\nPredicted AQI in India: {predicted_aqi[0]:.2f}")
print(f"Predicted AQI in United States of America: {predicted_aqi[1]:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual AQI', y='Predicted AQI', data=results_df, color='blue', label='Predictions')
plt.plot([results_df['Actual AQI'].min(), results_df['Actual AQI'].max()],
         [results_df['Actual AQI'].min(), results_df['Actual AQI'].max()],
         color='red', linestyle='--', label='Perfect Prediction')

plt.title('Actual vs Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.legend()
plt.tight_layout()
plt.show()

print("The Accuracy of the model is: ",{r2})