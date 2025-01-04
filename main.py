from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from analysis_app import process_datasets, create_dataset, train_lstm_model, predict_future

app = Flask(__name__, static_folder='static')
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
datasets = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_files():
    global datasets
    for file in request.files.getlist('files'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        datasets[file.filename] = pd.read_csv(filepath)
    return jsonify({'datasets': list(datasets.keys())})


# Load datasets into a dictionary
datasets_dict = {
    'SP500': pd.read_csv('SP500 (2).csv'),
    'FEDFUNDS': pd.read_csv('FEDFUNDS (2).csv'),
    'DJIA': pd.read_csv('DJIA (2).csv'),
    'NASDAQCOM': pd.read_csv('NASDAQCOM (2).csv'),
    'NASDAQ100': pd.read_csv('NASDAQ100 (1).csv'),
    'GDPC1': pd.read_csv('GDPC1.csv'),
    'UNRATE': pd.read_csv('UNRATE.csv'),
    'DCOILWTICO': pd.read_csv('DCOILWTICO.csv'),
    'DGS10': pd.read_csv('DGS10.csv'),
    'M2V': pd.read_csv('M2V.csv'),
    'WM2NS': pd.read_csv('WM2NS.csv'),
}

def preprocess_datasets(selected_datasets):
    merged_data = datasets_dict[selected_datasets[0]]
    for dataset_name in selected_datasets[1:]:
        merged_data = pd.merge(merged_data, datasets_dict[dataset_name], on='DATE', how='outer')

    merged_data.rename(columns={'DATE': 'Date'}, inplace=True)
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])

    mask = (merged_data['Date'] >= '1960-01-01') & (merged_data['Date'] <= '2024-12-31')
    filtered_data = merged_data.loc[mask].reset_index(drop=True)

    for column in filtered_data.columns[1:]:
        filtered_data[column] = pd.to_numeric(filtered_data[column], errors='coerce')

    filtered_data.fillna(method='ffill', inplace=True)
    filtered_data.dropna(inplace=True)

    return filtered_data

def create_dataset(data, selected_columns, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[selected_columns])
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step])
        y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(y), scaler

def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)
    return model

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        request_data = request.json
        selected_datasets = request_data.get('datasets', [])
        if not selected_datasets:
            return jsonify({'error': 'No datasets selected'}), 400

        # Check if datasets exist
        if not all(dataset in datasets_dict for dataset in selected_datasets):
            return jsonify({'error': 'One or more selected datasets are invalid'}), 400

        # Process datasets
        filtered_data = preprocess_datasets(selected_datasets)

        return jsonify({'message': 'Datasets processed successfully'})
    except ValueError as ve:
        print(f"Validation error in /analyze: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({'error': 'An internal error occurred during analysis'}), 500

@app.route('/available-datasets', methods=['GET'])
def available_datasets():
    return jsonify({'datasets': list(datasets_dict.keys())})

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.json
        selected_datasets = request_data.get('datasets', [])
        time_step = int(request_data.get('timeStep', 60))
        prediction_weeks = int(request_data.get('predictionWeeks', 8))

        if not selected_datasets:
            return jsonify({'error': 'No datasets selected'}), 400

        # Check if datasets exist
        if not all(dataset in datasets_dict for dataset in selected_datasets):
            return jsonify({'error': 'One or more selected datasets are invalid'}), 400

        # Process datasets
        filtered_data = preprocess_datasets(selected_datasets)

        # Create training dataset
        X, y, scaler = create_dataset(filtered_data, selected_datasets, time_step)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if len(X_train) == 0 or len(X_test) == 0:
            return jsonify({'error': 'Not enough data to train or test the model'}), 400

        # Train LSTM model
        model = train_lstm_model(X_train, y_train)

        # Generate predictions
        predictions = []
        last_sequence = X_test[-1]
        for _ in range(prediction_weeks):
            prediction = model.predict(np.expand_dims(last_sequence, axis=0))[0, 0]
            predictions.append(prediction)
            next_sequence = np.append(last_sequence[1:], [[prediction]], axis=0)
            last_sequence = next_sequence

        # Inverse transform predictions to original scale
        predictions_actual = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        labels = [f"Week {i + 1}" for i in range(prediction_weeks)]

        return jsonify({'labels': labels, 'predictedPrices': predictions_actual.tolist()})
    except ValueError as ve:
        print(f"Validation error in /predict: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': 'An internal error occurred during prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)
