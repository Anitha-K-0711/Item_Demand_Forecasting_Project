# Item_Demand_Forecasting_Project
![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/48ff415f-caf7-44b7-87cf-75448e59b563)

## Introduction
DateTime series data is often used in tracking industrial and business metrics. A datetime series is a set of observations recorded at different date and time points, 
whose value depends on the date and time it is recorded at. Since datetime runs forward, datetime series observations has a natural ordering.

Demand forecasts are fundamental to plan and deliver products and services. Accurate forecasting of demand can help the manufacturers 
to maintain appropriate stock which results in reduction in loss due to product not being sold and also reduces the opportunity cost.

In this project, historical sales data corresponding to multiple(50) items sold in 10 stores from the period of 2013 to 2017 is used to train and 
evaluate the performance of the machine learning models. The analysis of the model will be done to come up with the best model and r2_score metric is used 
to evaluate the model's performance.

The main objective of the project is to develop best ML model the accurately predicts the demand for next 3 months at the item level.

The built ML model is also deployed in streamlit. This streamlit app is a item demand forecaster where, when the manufacturer select the item number 
and date and click on the [Predict] button, the demand of the selected item 90 days from the selected date is calculated and displayed. 
Along with the prediction result, the details corresponding to the selected date is also displayed in the app for reference.

APP LINK: 

## Project Approach

### 1. Importing and Installing Necessary Libraries
I have imported necessary libraries like numpy, pandas, sklearn, seaborn, matplotlib, etc to use in the code block

### 2. Insights of the Data
I have loaded the dataset raw_data.csv and converted it to a dataframe using pandas. After loading the data, I took insights of all the features and target and explained in detail about each feature

### 3. Generic Cleaning of Data
I have used functions like null_values, dtypes, drop_duplicates to do general overall cleaning of the data

### 4. Deriving new columns from 'date' column
I have grouped the data based on 'item' and 'date' and aggregated the 'sales' column by addition. By doing this step, The stores column is getting ignored abruptly.

![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/69ef5db4-debc-4658-b34d-04d17a96807b)

After that, new columns out of 'date' columns like 'day', 'year', 'month', 'quarter', etc are derived and it is converted to integer datatype.

![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/32740a72-b147-49ad-8224-3801f3e05929)

### 5. Preparing 90 days rolling window data
I prepared a 90 day rolling window data for each item in the 'item' column using ".rolling()" function. By doing this step, the target column with 
each row containg cumulative sum of 90 days sales is successfully created.

![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/73667287-936c-4e66-96e6-a8c4344f535c)

### 6. Exploratory Data Analysis

Distribution of target plot, Feature v/s Target plots has been plotted using matplotlib and seaborn.

I observed the following from the EDA plots:

a) With respect to date column, the target have slightly upward trending cyclic pattern

b) With respect to year column, the number of items sold is increasing linearly with increase in year

c) The number of items sold is very high in the months of May, June, July, August. And, it is lowest in the months of Jan, Feb and Dec

d) The number of items sold are approximately the same in all the days of the week

e) The number of items sold are high in 2nd and 3rd quarter of each year than in the 1st and 4th quarter

### 7. Splitting the data

Since it is a datetime series data, it cannot be randomly split using train_test_split function. 

The first four years of the data is allocated to train and the last one year data is allocated to test data.

![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/ee3b8051-4b5d-4883-933e-dccecb40914b)

![image](https://github.com/Anitha-K-0711/Item_Demand_Forecasting_Project/assets/115402011/87199956-ddf0-4443-9fc1-af7b84363c1a)

### 8. Fit ML models

I have used 6 regression models and trained the data. The 6 models are,

1. Linear Regression
2. K-Nearest Neigbors
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. Extreme Gradient Boosting

For each model, both item-wise model training and overall data model training is performed. 

All the models have been evaluated using the metric r2_score. For item-wise training I also have calculated the average r2_score of all the 50 items for each model. 

Out of the 6 models, Extreme Gradient Boosting proved to be the best fit model with highest r2_score of 0.85 for item_wise model training and 
highest r2_score of 0.96 for overall data model training.

Refer item_demand_forecasting.ipynb for the code block of the above steps

## Deployment

This app is the fundamental to plan and deliver products and services, especially FMCG goods. Accurate forecasting of demand can 
help the manufacturers to maintain appropriate stock which results in reduction in loss due to product not being sold and also reduces the oppurtunity cost.

Once the manufacturer select the item number and date and click on the [Predict] button, the demand of the selected item 90 days from 
the selected date is calculated and displayed. Along with the prediction result, the details corresponding to the selected date is also 
displayed in the app for reference. By leveraging machine learning capabilities, A model that gives the best prediction to the manufacturer is established.

Refer main.py to view the code block for app deployment

App Link: 

## Further Scope

The current project has successfully built and evaluated a machine learning model to predict the demand for next 3 months at the item level. 
However there is still room for improvement and further scope in this project which includes,

1. Building a ML model with most important features: The data can be further trained with the ML models with only most important features of the data
   to check whether the results are getting fine tuned
   
2. Exploratory Data Analysis (EDA): In detail EDA is further required to the data in order to further understand and train the models to the data
   
3. Deployment: The plots of each feature and target need to be deployed in the streamlit app so that, the manufacturer will have better visualized
   ideas while predicting the demand
   
4. Model Comparison: In addition to the models evaluated in this project, other regression models could also be implemented and compared to
   identify the best performing model for this problem
   
5. Regular Maintenance: As the datetime series database grows and changes, the model's performance might degrade.
   Regular monitoring and maintenance of the model are necessary to ensure it continues to perform effectively

6. Deriving further columns: With respect to item and date column, daily sales for the week adding up to 'weekly sales',
   which in turn add up to 'monthly sales' and so on can be derived and can be used in model training to attain higher accuracy


