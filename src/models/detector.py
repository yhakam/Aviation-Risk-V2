import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES = ["speed_kmh","baro_altitude","vertical_rate_abs","altitude_diff","true_track","lat_bin","lon_bin"] # Features numériques

def run(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X, features_used = _prepare_features(df)
    X_scaled, scaler = _scale_features(X)
    df, model = _fit_model(df, X_scaled)
    _save_artifacts(model, scaler, features_used)

    return df

def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features_used = [col for col in FEATURES if col in df.columns]
    if not features_used:
        raise ValueError("Aucune feature disponible pour entraîner le modèle.")
    X = df[features_used].copy()
    X = X.fillna(X.median())
    return X, features_used

def _scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X),columns=X.columns,index=X.index)
    return X_scaled, scaler

def _fit_model(df: pd.DataFrame, X_scaled: pd.DataFrame) -> tuple[pd.DataFrame, IsolationForest]:
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(X_scaled)
    raw_scores = model.decision_function(X_scaled)
    min_s, max_s = raw_scores.min(), raw_scores.max()

    if max_s > min_s:
        df["anomaly_score"] = ((1 - (raw_scores - min_s) / (max_s - min_s)) * 100).round(1) #normalisation min-max
    else:
        df["anomaly_score"] = 0
    df["anomaly_flag"] = (model.predict(X_scaled) == -1).astype(int)
    if "risk_level" in df.columns:
        rule_high = df["risk_level"].isin(["HIGH", "MEDIUM"]).astype(int)
        agreement = (df["anomaly_flag"] == rule_high).mean()
        print(f"Accord rule-based / Isolation Forest : {agreement:.1%}")

    return df, model

def _save_artifacts(model: IsolationForest, scaler: StandardScaler, features_used: list[str],) -> None:
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(features_used, "models/features_used.pkl")

def save(df: pd.DataFrame) -> str:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    path = "data/processed/benchmark.csv"
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    df = pd.read_csv("data/processed/scored.csv")
    df = run(df)
    path = save(df)

    print(f"Benchmark saved -> {path}")