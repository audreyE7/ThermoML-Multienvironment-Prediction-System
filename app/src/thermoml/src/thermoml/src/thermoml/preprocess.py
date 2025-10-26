import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .features import add_dimensionless, FEATURE_COLUMNS, TARGET_COLUMN

def load_and_featurize(csv_path: str):
    df = pd.read_csv(csv_path)
    df = add_dimensionless(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return df, X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return (Xtr_s, Xte_s, ytr, yte, scaler)
