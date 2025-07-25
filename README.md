\# Air Quality Index (AQI) Prediction



This project implements a machine learning pipeline to predict the Air Quality Index (AQI) based on date and country input. The pipeline uses Random Forest Regression and includes preprocessing steps to extract time-based features and encode categorical variables.



\## Project Structure



\- `project.py`: Contains the full code for data preprocessing, model training, evaluation, and inference.

\- `data\_date.csv`: Dataset used for training and testing the model, containing date, country, and AQI values.



\## Features



1\) Converts date strings to datetime objects and extracts features such as year, month, day, day of week, and day of year.



2\) Uses one-hot encoding for the country column.



3\) Trains a RandomForestRegressor using scikit-learn's pipeline.



4\) Outputs performance metrics including MAE, RMSE, and R².



5\) Demonstrates predictions for custom date and country inputs.





\#Prerequisites



Ensure the following libraries are installed:



1)import pandas as pd



2\) from sklearn.model\_selection import train\_test\_split



3\) from sklearn.preprocessing import OneHotEncoder



4\) from sklearn.compose import ColumnTransformer



5\) from sklearn.pipeline import Pipeline



6\) from sklearn.ensemble import RandomForestRegressor



7\) from sklearn.metrics import mean\_absolute\_error, mean\_squared\_error, r2\_score



8\) import numpy as np



\##bash

pip install pandas scikit-learn numpy



