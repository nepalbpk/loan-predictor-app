# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:28:01 2025

@author: anupa
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create sample data
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, 1000),
    'credit_score': np.random.normal(680, 50, 1000),
    'age': np.random.randint(21, 65, 1000),
    'loan_amount': np.random.normal(15000, 4000, 1000)
})

X = data[['income', 'credit_score', 'age']]
y = data['loan_amount']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'C:\\Users\\anupa\\OneDrive\\Documents\\loan_predictor_app\loan_model.pkl')
joblib.dump(X.columns.tolist(), 'C:\\Users\\anupa\\OneDrive\\Documents\loan_predictor_app\\loan_model_features.pkl')
