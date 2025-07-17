import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from datetime import timedelta

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv', parse_dates=['date'])
    return df

df = load_data()

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("📊 Retail Sales Forecasting Dashboard")

# Sidebar Filters
store_ids = df['store_nbr'].unique()
product_families = df['family'].unique()

selected_store = st.sidebar.selectbox("Select Store", store_ids)
selected_family = st.sidebar.selectbox("Select Product Family", product_families)

model_choice = st.sidebar.radio("Choose Forecast Model", ("Prophet", "XGBoost"))

# Filtered data
filtered_df = df[(df['store_nbr'] == selected_store) & (df['family'] == selected_family)]

# Group and aggregate
daily_sales = filtered_df.groupby("date")["sales"].sum().reset_index()

# Display raw chart
st.subheader(f"🛒 Sales Data for Store {selected_store} - {selected_family}")
st.line_chart(daily_sales.set_index("date"))

# Forecasting
n_days = st.slider("Select number of days to forecast", min_value=15, max_value=90, value=30)

if model_choice == "Prophet":
    prophet_df = daily_sales.rename(columns={"date": "ds", "sales": "y"})

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)

    st.subheader("📈 Forecast with Prophet")
    fig1 = px.line(forecast, x="ds", y="yhat", title="Forecasted Sales")
    fig1.add_scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual")
    st.plotly_chart(fig1, use_container_width=True)

    # Evaluation
    merged = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='inner')
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    r2 = r2_score(merged['y'], merged['yhat'])

elif model_choice == "XGBoost":
    df_xgb = daily_sales.copy()
    df_xgb['day'] = df_xgb['date'].dt.day
    df_xgb['month'] = df_xgb['date'].dt.month
    df_xgb['year'] = df_xgb['date'].dt.year

    df_xgb.set_index('date', inplace=True)
    train = df_xgb.iloc[:-n_days]
    test = df_xgb.iloc[-n_days:]

    features = ['day', 'month', 'year']
    X_train, y_train = train[features], train['sales']
    X_test, y_test = test[features], test['sales']

    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    forecast_df = test.reset_index()
    forecast_df['forecast'] = preds

    st.subheader("📈 Forecast with XGBoost")
    fig2 = px.line(forecast_df, x="date", y="forecast", title="Forecasted Sales")
    fig2.add_scatter(x=forecast_df["date"], y=forecast_df["sales"], name="Actual")
    st.plotly_chart(fig2, use_container_width=True)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

# Metrics
st.markdown("### 🧮 Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# Download Forecast
st.markdown("### 📥 Download Forecast")
if model_choice == "Prophet":
    output_df = forecast[['ds', 'yhat']].rename(columns={"ds": "Date", "yhat": "Predicted Sales"})
else:
    output_df = forecast_df[['date', 'forecast']].rename(columns={"date": "Date", "forecast": "Predicted Sales"})

csv = output_df.to_csv(index=False).encode()
st.download_button("Download Forecast as CSV", data=csv, file_name='forecast.csv', mime='text/csv')
