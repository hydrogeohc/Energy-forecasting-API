# Energy consumption forecasting

## Overview

This project focuses on modeling and predicting energy consumption using a variety of machine-learning techniques. It involves loading, preprocessing, and exploring energy consumption data, engineering relevant features, building predictive models, and evaluating their performance. The project also implements a Flask API to serve the model evaluation results.

## File Structure

├── app.py                       # Flask application for serving model results via API
├── data_preprocessing.py        # Data loading, preprocessing, feature engineering, and visualization
├── model_building.py            # Model training, evaluation (including LSTM), and time series cross-validation
├── energy_consumption_data.csv  # The dataset used for training and evaluation
├── requirements.txt             # List of Python dependencies for the project
└── README.md                    # This file (project overview and setup instructions)



## Dependencies

The project relies on the following Python libraries:

*   pandas: For data manipulation and analysis
*   scikit-learn: For various machine learning models and evaluation metrics
*   tensorflow/keras: For building and training the LSTM model
*   Flask: For creating the web API
*   numpy: For numerical computations

To install all required dependencies, it is recommended to use a virtual environment and run:

pip install -r requirements.txt

A sample `requirements.txt` is shown below:

Flask==3.1.0
pandas==2.2.3
scikit-learn==1.6.1
tensorflow==2.18.0
numpy==2.02
seaborn==0.13.2
matplotlib==3.10.1

## Usage

1.  **Clone the repository:**

    ```
    git clone https://github.com/hydrogeohc/Optim_Energy
    cd Optim_Energy
    ```

2.  **Create a virtual environment:**

    ```
    python3 -m venv venv  # create the virtual environment
    source venv/bin/activate   # activate on macOS/Linux
    # venv\Scripts\activate   # activate on Windows
    ```

3.  **Install dependencies:**

    ```
    pip install --upgrade pip # upgrade pip
    pip install -r requirements.txt # Install the dependencies
    ```

4.  **Run the Flask API:**

    ```
    python app.py
    ```

5.  **Access the API endpoints:**

    *   `http://127.0.0.1:5000/` - Returns a welcome message.
    *   `http://127.0.0.1:5000/model_results` - Retrieves model evaluation results in JSON format. *Note:* The AirPlay Receiver service on macOS may conflict with port `5000`. If you encounter issues, try changing the Flask port to `5001` in `app.py` and accessing `http://127.0.0.1:5001/model_results`.

## Models

The following machine learning models are implemented and evaluated in this project:

*   Linear Regression
*   Random Forest
*   Gradient Boosting
*   LSTM (Long Short-Term Memory)

*Note*: The traditional models (Linear Regression, Random Forest, and Gradient Boosting) are evaluated using Time Series Cross-Validation. However, the current implementation evaluates the LSTM using a single train/test split. The LSTM results are therefore not directly comparable to the other models.

## Data

The `energy_consumption_data.csv` file contains the historical energy consumption data used to train and evaluate the models. The dataset includes features such as:

*   `timestamp`: Date and time of the reading
*   `temperature`: Temperature at the time of the reading
*   `humidity`: Humidity at the time of the reading
*   `energy_consumption`: The target variable, representing the energy consumed

The project also engineers new features such as:

*   `temp_humidity_interaction`: Interaction term between temperature and humidity
*   `energy_consumption_lag1`: Energy consumption from the previous time period (lag 1)
*   `energy_consumption_lag2`: Energy consumption from two time periods prior (lag 2)
*   `hour`, `day_of_week`, `is_weekend`: Temporal features derived from the timestamp.

## Evaluation Metrics

The models are evaluated based on the following metrics:

*   Mean Squared Error (MSE)
*   Mean Absolute Error (MAE)
*   Bias

## Troubleshooting

*   **ModuleNotFoundError:** Ensure that all Python dependencies are installed in your virtual environment. If you see the error, install the libraries that are missing.
*   **File Permissions:** Make sure that the `energy_consumption_data.csv` file is readable by the Flask application.


## License

MIT License