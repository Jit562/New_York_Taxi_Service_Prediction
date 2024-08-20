# Project Name - NYC taxi trip time Prediction: Predicting total ride duration of taxi trips in New York City
* Project Type - Regression/Supervised Learning
* Contribution - Individual
* Team Member - 1
* BY-Jitendra Prasad (jitendra.mits2@gmail.com)

**Project Summary** - The NYC Taxi Time Prediction project aims to predict the amount of time a taxi trip will take in New York City, based on various features such as pickup and dropoff locations, time of day, and weather conditions.

A regression model was developed to predict the duration of the taxi trip. The model was trained on a large dataset of over 1.5 million taxi trips, which were randomly split into training and testing sets.

The features used in the regression model included distance, pickup and dropoff coordinates, pickup datetime, day of the week, and weather conditions such as temperature, precipitation, and wind speed.

The model was evaluated using various metrics such as Mean Squre Error (MSE) and Root Mean Squared Error (RMSE),R2 Score,Adjusted R2-Score and was compared to other machine learning algorithms such as Linear Regression,Decision Tree , Gradient Boosting. The regression model outperformed the other algorithms in terms of accuracy, with an R2 score of 62%.

Overall, the NYC Taxi Time Prediction project demonstrates the potential for regression models to accurately predict the duration of taxi trips in New York City, using a combination of various features such as location, time, and distance.

**NYC Taxi Dataset**

The NYC Taxi dataset contains detailed trip records, offering a comprehensive view of taxi operations across the city. The dataset includes key information such as pickup and dropoff locations (latitude and longitude or taxi zones), pickup and dropoff times, trip distances, passenger counts, and fare details (including tips, tolls, and total amounts). This data is invaluable for analyzing traffic patterns, understanding demand fluctuations, and optimizing taxi and ride-sharing operations.


**Trip Information:**

**Pickup and Dropoff Datetimes:** Timestamps indicating when and where the ride started and ended.

**Pickup and Dropoff Locations:** Latitude and longitude coordinates, or taxi zones indicating where the ride began and ended.

**Passenger Count:** The number of passengers in the taxi during the ride.

**Trip Distance:** The distance covered during the trip, typically in miles.

**Other Metadata:**

**Vendor ID:** Identifies the provider of the taxi service.

**Trip Type:** Indicates whether the trip was a street-hail or a dispatch trip.

**Store and Forward Flag:** Indicates whether the trip record was held in the taxi’s memory before sending to the vendor (for instances when the vehicle was outside network coverage).

**Conclusion for Model Training:**

*  There were a lot of outliers in our variables some values were near to zero, we tried to remove those values but we found that we were losing a lot of data. we trained our model using various algorithms and we got an accuracy of 62%.

*  we were curious whether the model was overfit or not, hopefully it was not, as it gave pretty much similar results for train and test data in all the algorithms tried.

*   In all the above model's graph we saw that actual and predicted values are almost near to each other (lines coinciding) in only 2 models namely: Decission Tree and Gradiant Boost regresion. R2 scores were also high for the above two models and MSE scores were also low in these models which satisafies the requirements of a good model.

*  So we came to a conclusion that removing data removes a lot of information, new column if highly collinear can give pseudo good results,also we got our best R2 score from GradiantBoostregresion* model,we tried taking an optimum parameter so that our model doesnt overfit.

**Use This Project:**

* **step-1:** Clone the Repo
* **step-2:** Import Libraries

* import pikle
* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* import seaborn as sns
* import warnings
* warnings.filterwarnings('ignore')
* import math

* from sklearn.model_selection import train_test_split # feature selection
* from sklearn.compose import ColumnTransformer # data preprocessing
* from sklearn.pipeline import Pipeline # Ml Pipe line
* from sklearn.preprocessing import OneHotEncoder # Categorical data preprocessing
* from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler # Numerical data preprocessing
* from sklearn.preprocessing import PolynomialFeatures # Numerical data preprocessing

* from sklearn.linear_model import Lasso, Ridge ## For handling overfitting model
* from sklearn.linear_model import LinearRegression
* from sklearn.tree import DecisionTreeRegressor ## Tree model
* from sklearn.ensemble import GradientBoostingRegressor


* from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,mean_absolute_percentage_error #Evaluation Matrix
* from sklearn.model_selection import GridSearchCV #Hyperparameter Tuning

* **step-3:** Load Dataset
  
* **Run command**

  # To load the model 
* with open('model_gradient_boost.pkl', 'rb') as file:
    * loaded_model = pickle.load(file) 

 # To load the data
* with open('model_data.pkl', 'rb') as file:
    * loaded_model = pickle.load(file)      


    


