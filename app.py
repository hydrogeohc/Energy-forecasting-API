from flask import Flask, jsonify
import pandas as pd
from data_preprocessing import preprocess_and_explore_data  # Import from data_preprocessing.py
from model_building import train_and_evaluate_models  # Import from model_building.py

app = Flask(__name__)

@app.route('/model_results', methods=['GET'])
def get_model_results():
    """
    API endpoint to preprocess data, train models, and return their evaluation results.
    """
    try:
        # Step 1: Load and preprocess data
        file_path = './energy_consumption_data.csv'  # Path to the dataset
        preprocessed_df = preprocess_and_explore_data(file_path)

        # Step 2: Train models and evaluate performance
        model_results = train_and_evaluate_models(preprocessed_df)

        # Step 3: Return results as JSON
        return jsonify(model_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
