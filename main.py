import datetime
import pandas as pd
import streamlit as st
import numpy as np
from xgboost import XGBRegressor

train_df1 = pd.read_csv('train_df1.csv')
test_df1 = pd.read_csv('test_df1.csv')

x_train = train_df1[
    ['item', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'days_in_month', 'quarter', 'is_leap_year',
     'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']].values
y_train = train_df1['target'].values
x_test = test_df1[
    ['item', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 'days_in_month', 'quarter', 'is_leap_year',
     'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end']].values
y_test = test_df1['target'].values

model_xgb = XGBRegressor()
model_xgb.fit(x_train, y_train)

model_xgb.save_model('xgb_model.json')

model = XGBRegressor()
model.load_model('xgb_model.json')


st.title('Item Demand Forecasting')

with st.expander('About the App'):
    st.markdown(
        '<div style="text-align: justify;">This app is the fundamental to plan and deliver products and services, '
        'especially FMCG goods. Accurate forecasting of demand can help the manufacturers to maintain appropriate '
        'stock which results in reduction in loss due to product not being sold and also reduces the oppurtunity cost.</div>',
        unsafe_allow_html=True)
    st.write(" ")
    st.markdown(
        '<div style="text-align: justify;"> Once the manufacturer select the item number and date and click on the [Predict] button, '
        'the demand of the selected item 90 days from the selected date is calculated and displayed. Along with the prediction result, '
        'the details corresponding to the selected date is also displayed in the app for reference. By leveraging machine learning '
        'capabilities, A model that gives the best prediction to the manufacturer is established.</div>',
        unsafe_allow_html=True)
    st.write(" ")

st.header('Please fill the following details:')

with st.form('Please fill the following details:'):
    item = st.number_input('item', min_value=1, max_value=50, value=1)
    d = st.date_input('Date please:', min_value=datetime.date(2018, 1, 1), max_value=datetime.date(2030, 12, 31))

    dates = pd.to_datetime(d)

    year = dates.year
    month = dates.month
    day = dates.day
    day_of_week = dates.day_of_week
    day_of_year = dates.day_of_year
    days_in_month = dates.days_in_month
    quarter = dates.quarter
    is_leap_year = dates.is_leap_year
    is_month_start = dates.is_month_start
    is_month_end = dates.is_month_end
    is_quarter_start = dates.is_quarter_start
    is_quarter_end = dates.is_quarter_end
    is_year_start = dates.is_year_start
    is_year_end = dates.is_year_end

    submitted = st.form_submit_button('Predict')

    if submitted:
        st.write("year: ", dates.year, "month: ", dates.month, "day: ", dates.day, "quarter: ", dates.quarter)
        st.write("day_of_year: ", dates.day_of_year, "days_in_month: ", dates.days_in_month, "is_leap_year: ", dates.is_leap_year)
        st.write("is_month_start: ", dates.is_month_start, "is_month_end: ", dates.is_month_end)
        st.write("is_quarter_start: ", dates.is_quarter_start, "is_quarter_end: ", dates.is_quarter_end)
        st.write("is_year_start: ", dates.is_year_start, "is_year_end: ", dates.is_year_end)
        st.write(" ")

        result = model.predict(np.array(
            [item, year, month, day, day_of_week, day_of_year, days_in_month, quarter, is_leap_year, is_month_start,
             is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end]).reshape(1, -1))
        st.write("The demand of the selected item 90 days from the selected date is: ")
        st.success(*result)
