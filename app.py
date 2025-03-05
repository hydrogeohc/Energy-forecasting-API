from flask import Flask, jsonify
import pandas as pd
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_preprocessing import preprocess_and_explore_data
from model_building import train_and_evaluate_models

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Energy Consumption Model API. Use /model_results to get model evaluation results."

@app.route('/model_results', methods=['GET'])
def get_model_results():
    """
    API endpoint to preprocess data, train models, and return their evaluation results.
    """
    try:
        # Step 1: Load and preprocess data
        file_path = os.path.join(os.path.dirname(__file__), 'energy_consumption_data.csv')
        preprocessed_df = preprocess_and_explore_data(file_path)

        # Step 2: Train models and evaluate performance
        model_results = train_and_evaluate_models(preprocessed_df)

        # Step 3: Return results as JSON
        return jsonify(model_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(error):
    return jsonify({"error": "Page not found. Please check the URL and try again."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)  # Use port 5001 instead of 5000
