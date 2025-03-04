from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sys
sys.path.append('./data_processing')
from data_preprocessing import engineer_features

def train_and_evaluate_models(df):
    # Engineer features
    df = engineer_features(df)
    
    # Define features (X) and target (y)
    X = df[['temperature', 'humidity', 'temp_humidity_interaction']]
    y = df['energy_consumption']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate traditional models
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {"MSE": mse, "RMSE": rmse, "R2": r2}
    
    # Prepare data for LSTM
    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)
    
    # Create and train LSTM model
    lstm_model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X.shape[1])),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
    
    # Evaluate LSTM model
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
    r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
    results["LSTM"] = {"MSE": mse_lstm, "RMSE": rmse_lstm, "R2": r2_lstm}
    
    return results

if __name__ == '__main__':
    # Load your dataset here (assuming it's already preprocessed)
    df = pd.read_csv('./energy_consumption_data.csv')
    
    # Train models and evaluate their performance
    model_results = train_and_evaluate_models(df)
    
    # Print the results for each model
    for model_name, metrics in model_results.items():
        print(f"{model_name}:")
        print(f" MSE: {metrics['MSE']:.2f}")
        print(f" RMSE: {metrics['RMSE']:.2f}")
        print(f" R2: {metrics['R2']:.2f}")
