import streamlit as st
import pandas as pd
import joblib

results = joblib.load("full_pipeline_diabetes.pkl")
pipeline = results['pipeline']

st.title("Prédiction du Diabète")


pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 300, 120)
bloodpressure = st.number_input("BloodPressure", 0, 200, 70)
skinthickness = st.number_input("SkinThickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 1000, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", 0.0, 5.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

if st.button("Prédire"):
    X_new = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness,
                           insulin, bmi, diabetes_pedigree, age]],
                         columns=pipeline.feature_names_in_)
    pred = pipeline.predict(X_new)
    proba = pipeline.predict_proba(X_new)

    st.write("Résultat :", "Diabète" if pred[0] == 1 else "Pas de diabète")
    st.write("Probabilité :", proba[0])


