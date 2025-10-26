import argparse, os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from .preprocess import load_and_featurize, split_and_scale

def train(data_csv: str, out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    df, X, y = load_and_featurize(data_csv)
    Xtr_s, Xte_s, ytr, yte, scaler = split_and_scale(X, y)

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtr_s, ytr)

    preds = model.predict(Xte_s)
    mae = mean_absolute_error(yte, preds)
    r2  = r2_score(yte, preds)
    print(f"MAE: {mae:.2f}  |  R2: {r2:.3f}")

    joblib.dump(model, os.path.join(out_dir, "thermoml_rf.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    print(f"Saved model + scaler to {out_dir}/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_thermal_dataset.csv")
    ap.add_argument("--out",  default="artifacts")
    args = ap.parse_args()
    train(args.data, args.out)
