import streamlit as st 
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Streamlit setup
st.set_page_config(page_title="Time Series Forecasting App", layout="centered")
st.title("📈 Time Series Forecasting App")

# Pre-defined datasets
datasets = {
    "Airline Passengers": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    "US Macro Monthly": "https://raw.githubusercontent.com/PJalgotrader/Deep_forecasting-USU/main/Platforms%20and%20tools/streamlit/US_macro_monthly.csv",
}

# Dataset selection
dataset_name = st.selectbox("Choose a dataset", ["Upload your own CSV/Excel"] + list(datasets.keys()))

def load_data():
    if dataset_name == "Upload your own CSV/Excel":
        uploaded = st.file_uploader("Upload your time series file", type=["csv", "xlsx"])
        if uploaded:
            return pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    else:
        url = datasets[dataset_name]
        return pd.read_csv(url)

df = load_data()
if df is not None:
    st.write(df.head())
    target_col = st.selectbox("Select the target variable", df.select_dtypes(include=np.number).columns)
    time_col = st.selectbox("Select the time column", df.columns)

    # Preprocessing
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)
    
    # Handling missing values
    imputer = SimpleImputer(strategy="mean")
    df[target_col] = imputer.fit_transform(df[[target_col]])
    
    # Detecting and handling outliers using Z-score
    threshold = 3  # Z-score threshold for outlier detection
    z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
    df = df[z_scores < threshold]
    
    # Standardizing the data (if needed for ML models)
    scaler = StandardScaler()
    df[target_col] = scaler.fit_transform(df[[target_col]])
    
    # Drop missing rows (if any after preprocessing)
    df = df.dropna()

    st.line_chart(df)

    # Split
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)
    train, val, test = df.iloc[:train_size], df.iloc[train_size:train_size+val_size], df.iloc[train_size+val_size:]

    model_type = st.selectbox("Choose a forecasting model", [
        "Holt-Winters", "ARIMA", "SARIMA", "SES", "Holt’s Linear Trend",
        "Random Forest", "XGBoost", "LightGBM", "SVR", "KNN", "Linear Regression"
    ])

    def evaluate(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return rmse, mae, mape

    def create_lag_features(df, lags=7):
        df_lag = df.copy()
        for i in range(1, lags + 1):
            df_lag[f"lag_{i}"] = df_lag[target_col].shift(i)
        return df_lag.dropna()

    if st.button("Run Forecast"):
        forecast, true_vals = None, test[target_col].values

        if model_type in ["Holt-Winters", "SES", "Holt’s Linear Trend"]:
            trend = "add" if model_type != "SES" else None
            model = ExponentialSmoothing(train[target_col], trend=trend, seasonal=None)
            fitted = model.fit()
            forecast = fitted.forecast(len(test))

        elif model_type == "ARIMA":
            model = ARIMA(train[target_col], order=(5, 1, 0))
            forecast = model.fit().forecast(steps=len(test))

        elif model_type == "SARIMA":
            model = SARIMAX(train[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            forecast = model.fit().forecast(steps=len(test))

        elif model_type in ["Random Forest", "XGBoost", "LightGBM", "SVR", "KNN", "Linear Regression"]:
            df_ml = create_lag_features(df)
            X = df_ml.drop(columns=[target_col])
            y = df_ml[target_col]

            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size+7:]
            y_test = y.iloc[train_size+7:]
            true_vals = y_test.values

            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100)
            elif model_type == "XGBoost":
                model = XGBRegressor(n_estimators=100)
            elif model_type == "LightGBM":
                model = lgb.LGBMRegressor(n_estimators=100)
            elif model_type == "SVR":
                model = SVR()
            elif model_type == "KNN":
                model = KNeighborsRegressor(n_neighbors=5)
            elif model_type == "Linear Regression":
                model = LinearRegression()

            model.fit(X_train, y_train)
            forecast = model.predict(X_test)

        # Evaluation
        rmse, mae, mape = evaluate(true_vals, forecast)
        st.subheader(f"Model: {model_type}")
        st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df[target_col], label="Original Data", color="blue")
        ax.plot(df.index[-len(forecast):], forecast, label="Forecast", color="red", linestyle="--")
        ax.set_title(f"Forecast with {model_type}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)
