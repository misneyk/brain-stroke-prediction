from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('stroke_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Extract features from form
    features = pd.DataFrame({
        'age': [float(data['age'])],
        'hypertension': [int(data['hypertension'])],
        'heart_disease': [int(data['heart_disease'])],
        'avg_glucose_level': [float(data['avg_glucose_level'])],
        'bmi': [float(data['bmi'])],
        'gender': [data['gender']],
        'ever_married': [data['ever_married']],
        'work_type': [data['work_type']],
        'Residence_type': [data['Residence_type']],
        'smoking_status': [data['smoking_status']]
    })
    
    # Predict
    prediction = model.predict(features)
    
    # Return prediction
    if prediction[0] == 1:
        result = 'High risk of stroke'
    else:
        result = 'Low risk of stroke'
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
