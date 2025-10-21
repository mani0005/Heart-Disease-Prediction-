from flask import Flask, render_template, request
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

# Load scaler
scaler = joblib.load('models/scaler.pkl')

# Load features
with open('models/features.json', 'r') as f:
    FEATURES = json.load(f)

# Load only KNN and Logistic Regression models
models = {
    'knn': ('K-Nearest Neighbors', joblib.load('models/knn_model.pkl')),
    'logreg': ('Logistic Regression', joblib.load('models/logreg_model.pkl'))
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    form_data = {}
    if request.method == 'POST':
        try:
            name = request.form.get('name', 'User').strip()
            form_data['name'] = name

            # Collect features from form
            input_values = []
            for feature in FEATURES:
                value = request.form.get(feature, '')
                form_data[feature] = value
                input_values.append(float(value))

            # Selected model key
            selected_model_key = request.form.get('model')
            form_data['model'] = selected_model_key

            if selected_model_key not in models:
                raise ValueError("Invalid model selected.")

            model_name, model = models[selected_model_key]

            # Prepare input for prediction
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][pred]

            # Prepare response messages
            if pred == 0:
                prediction = f"{name}, you are safe. 😊 (Model: {model_name})"
            else:
                prediction = f"{name}, you are at risk. 👽 (Model: {model_name})"

            confidence = f"{prob * 100:.2f}%"

        except Exception as e:
            prediction = f"Error: {str(e)}"
            confidence = None

    return render_template(
        "index.html", 
        models={k: v[0] for k, v in models.items()}, 
        prediction=prediction, confidence=confidence, form_data=form_data
    )


if __name__ == '__main__':
    app.run(debug=True)
