# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:30:09 2025

@author: anupa
"""

from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
features = joblib.load("loan_model_features.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        income = float(request.form["income"])
        credit_score = float(request.form["credit_score"])
        age = float(request.form["age"])

        input_df = pd.DataFrame([{
            "income": income,
            "credit_score": credit_score,
            "age": age
        }]).reindex(columns=features)

        prediction = model.predict(input_df)[0]
    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
