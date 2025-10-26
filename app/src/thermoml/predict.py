import argparse, json, joblib, numpy as np, pandas as pd
from .features import add_dimensionless, FEATURE_COLUMNS

def predict_one(payload_path: str, model_path="artifacts/thermoml_rf.joblib", scaler_path="artifacts/scaler.joblib"):
    with open(payload_path, "r") as f:
        payload = json.load(f)  # dict of feature values

    df = pd.DataFrame([payload])
    df = add_dimensionless(df)
    X = df[FEATURE_COLUMNS]

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    Xs = scaler.transform(X)
    yhat = model.predict(Xs)[0]
    return float(yhat)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to JSON with feature values")
    ap.add_argument("--model", default="artifacts/thermoml_rf.joblib")
    ap.add_argument("--scaler", default="artifacts/scaler.joblib")
    args = ap.parse_args()
    y = predict_one(args.json, args.model, args.scaler)
    print(f"Predicted T_max: {y:.2f} °C")
