import streamlit as st, pandas as pd, joblib
from pathlib import Path
from src.thermoml.features import add_dimensionless, FEATURE_COLUMNS

st.title("ThermoML — Quick Predictor")

with st.form("inputs"):
    k = st.number_input("Thermal conductivity k [W/m·K]", 10.0, 300.0, 150.0)
    rho = st.number_input("Density ρ [kg/m³]", 500.0, 9000.0, 2700.0)
    cp = st.number_input("Specific heat c_p [J/kg·K]", 300.0, 1500.0, 900.0)
    eps = st.number_input("Emissivity ε [-]", 0.0, 1.0, 0.85)
    h = st.number_input("Convective h [W/m²·K]", 0.0, 500.0, 15.0)
    qin = st.number_input("Heat flux q_in [W/m²]", 0.0, 150000.0, 40000.0)
    Tenv = st.number_input("Environment temp [K]", 150.0, 400.0, 298.0)
    Lc = st.number_input("Characteristic length Lc [m]", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    t = st.number_input("Time t [s]", 1.0, 600.0, 60.0)
    env = st.selectbox("Environment", ["desert(0)","ocean(1)","vacuum(2)","space(3)"])
    env_code = int(env[-2])  # last char inside parens
    go = st.form_submit_button("Predict")

if go:
    df = pd.DataFrame([{
        "k":k,"rho":rho,"cp":cp,"epsilon":eps,"h":h,"q_in":qin,"T_env":Tenv,
        "Lc":Lc,"t":t,"env_code":env_code
    }])
    df = add_dimensionless(df)
    X = df[FEATURE_COLUMNS]

    model = joblib.load("artifacts/thermoml_rf.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")
    yhat = model.predict(scaler.transform(X))[0]
    st.success(f"Predicted T_max = **{yhat:.2f} °C**")
    st.write(df)
