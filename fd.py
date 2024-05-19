import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv('stroke_data.csv')

# Drop rows with missing values
data = data.dropna()

# Split dataset into features and target
X = data.drop('stroke', axis=1)
y = data['stroke']

# List of categorical features
cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# List of numerical features
num_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])

# Create pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=10000, random_state=42))])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'stroke_prediction_model.pkl')
