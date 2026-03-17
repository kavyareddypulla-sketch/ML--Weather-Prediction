# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the programe

2.Import the required libraries such as pandas and sklearn.

3.Load the environmental sensor dataset from the CSV file.

4.Select Humidity, WindSpeed and Pressure as input features.

5.Select Temperature, PM2.5 and Energy as output variables.

6.Split the dataset into training and testing data.

7.Initialize the Random Forest Regressor model.

8.Train the model using the training dataset

9.Predict the output values using the test dataset

10.Display the predicted values.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: KAVYA
RegisterNumber:  212225240110
*/
import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/acer/Downloads/weather-station-eee-block_2024_07_13.csv")
print(data.head())
print(data.isnull().sum())
data.fillna(method='ffill', inplace=True)
X = data[['hum', 'pressure', 'wind_speed']]
y_temp = data['tem']
y_pm = data['pm2_5']
y_energy = data['tsr']
from sklearn.model_selection import train_test_split

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=0)
X_train, X_test, y_pm_train, y_pm_test = train_test_split(X, y_pm, test_size=0.2, random_state=0)
X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

model_temp = RandomForestRegressor(n_estimators=100, random_state=0)
model_pm = RandomForestRegressor(n_estimators=100, random_state=0)
model_energy = RandomForestRegressor(n_estimators=100, random_state=0)
model_temp.fit(X_train, y_temp_train)
model_pm.fit(X_train, y_pm_train)
model_energy.fit(X_train, y_energy_train)
temp_pred = model_temp.predict(X_test)
pm_pred = model_pm.predict(X_test)
energy_pred = model_energy.predict(X_test)
from sklearn.metrics import mean_squared_error

print("Temperature MSE:", mean_squared_error(y_temp_test, temp_pred))
print("PM2.5 MSE:", mean_squared_error(y_pm_test, pm_pred))
print("Energy MSE:", mean_squared_error(y_energy_test, energy_pred))

```

## Output:
<img width="401" height="110" alt="image" src="https://github.com/user-attachments/assets/2d6bf58b-c993-459f-aa58-5f68bbf82084" />


## Result:
