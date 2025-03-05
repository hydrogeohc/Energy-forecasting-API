from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
import os

# Add the directory containing your data_preprocessing to the Python path
# This assumes data_preprocessing.py is in a directory called 'data_processing'
sys.path.append('./data_processing')  # Adjust if the path is different
from data_preprocessing import engineer_features

def calculate_bias(y_true, y_pred):
    """Calculates the bias (mean error) of the predictions."""
    return np.mean(y_pred - y_true)

def train_and_evaluate_models(df):
    """Trains and evaluates multiple models, including LSTM, using time series cross-validation.

    Adds lag features (energy_consumption_lag1 and energy_consumption_lag2)
    and calculates Mean Absolute Error (MAE) and Bias instead of R-squared.
    """

    # Engineer features (including lag features)
    df = engineer_features(df)

    # Drop rows with NaN values introduced by lag features.
    df = df.dropna() # Essential! Otherwise LinearRegression will fail.

    # Define features (X) and target (y)
    X = df[['temperature', 'humidity', 'energy_consumption_lag1','temp_humidity_interaction']]  #  features
    y = df['energy_consumption']

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits

    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {} #Initialize results
    for model_name in models:
        results[model_name] = {"MSE":[], "MAE":[], "Bias":[]} #Metrics will be lists now

    #Time series cross validation loop:
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train and evaluate traditional models
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred) # Added MAE
            bias = calculate_bias(y_test, y_pred) # Added bias

            results[model_name]["MSE"].append(mse)
            results[model_name]["MAE"].append(mae)
            results[model_name]["Bias"].append(bias)

    # Calculate the average metrics across all the folds
    for model_name in models:
        results[model_name]["MSE"] = np.mean(results[model_name]["MSE"])
        results[model_name]["MAE"] = np.mean(results[model_name]["MAE"])
        results[model_name]["Bias"] = np.mean(results[model_name]["Bias"])

    #LSTM - *NOT* using time series cross-validation - requires refactoring to use TSCV properly
    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

    # Create and train LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X.shape[1])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)

    # Evaluate LSTM model - USING a SINGLE train_test_split, *NOT* TSCV!
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm = y_pred_lstm.flatten()

    mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)  # Added MAE for LSTM
    bias_lstm = calculate_bias(y_test_lstm, y_pred_lstm)  # Added bias for LSTM

    results["LSTM"] = {"MSE": mse_lstm, "MAE": mae_lstm, "Bias": bias_lstm}  #

    return results

if __name__ == '__main__':
    # Load your dataset here (assuming it's already preprocessed)
    df = pd.read_csv('./energy_consumption_data.csv')

    # Train models and evaluate their performance
    model_results = train_and_evaluate_models(df)

    # Print the results for each model
    for model_name, metrics in model_results.items():
        print(f"{model_name}:")
        print(f"  MSE: {metrics['MSE']:.2f}")
        print(f"  MAE: {metrics['MAE']:.2f}")
        print(f"  Bias: {metrics['Bias']:.2f}")

