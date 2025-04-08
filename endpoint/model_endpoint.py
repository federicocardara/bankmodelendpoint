from flask import Flask, request, jsonify
import pickle
import numpy as np

# Cargar el modelo entrenado y el preprocesador desde la carpeta 'results'
with open('results/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('results/best_model_encoder.pkl', 'rb') as le_file:
    preprocessor = pickle.load(le_file)

# Inicializar la aplicación Flask
app = Flask(__name__)

# Define the expected feature names based on the dataset
expected_features = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud
    data = request.get_json(force=True)

    # Ensure the incoming data has the correct keys
    missing_features = [feature for feature in expected_features if feature not in data]
    if missing_features:
        return jsonify({
            'error': 'Missing one or more required features',
            'missing_features': missing_features
        }), 400

    # Prepare the input data: Ensure the JSON data matches the order of expected features
    input_data = np.array([[data[feature] for feature in expected_features]])

    # Preprocess the data (OneHotEncoding)
    input_data_preprocessed = preprocessor.transform(input_data)

    # Perform the prediction
    prediction = model.predict(input_data_preprocessed)

    # Return the result as a JSON object
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
