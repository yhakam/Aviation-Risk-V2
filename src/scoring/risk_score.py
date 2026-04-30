import pandas as pd
from pathlib import Path
import numpy as np


RULES = {
    "speed_anomaly": {"weight": 25, "label":"Vitesse anormale"},
    "altitude_anomaly": {"weight":25, "label":"Altitude anormale"},
    "vertical_anomaly": {"weight":20, "label": "Taux vertical extrême"},
    "altitude_diff": {"weight": 15, "label": "Ecart baro/géo élevé"},
    "vertical_rate_abs": {"weight": 15, "label": "Variation verticale forte"}
}

def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _score(df)
    df = _level(df)
    df = _justification(df)
    return df

def _score(df: pd.DataFrame) -> pd.DataFrame:
    df["risk_score"] = 0.0

    for col, rule in RULES.items():
        if col not in df.columns:
            continue
        if col in ["speed_anomaly","altitude_anomaly","vertical_anomaly"]:
            df["risk_score"] += df[col].fillna(0) * rule["weight"]

        elif col == "altitude_diff":
            max_val = df[col].abs().quantile(0.99)
            if pd.notna(max_val) and max_val >0:
                normalized = (df[col].abs() / max_val).clip(0, 1)
                df["risk_score"] += normalized.fillna(0) * rule ["weight"]
        elif col == "vertical_rate_abs":
            max_val = df[col].quantile(0.99)
            if pd.notna(max_val) and max_val > 0:
                normalized = (df[col] / max_val).clip(0,1)
                df["risk_score"] += normalized.fillna(0) * rule ["weight"]
    df["risk_score"] = df["risk_score"].clip(0, 100).round(1)    
    return df

def _level(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["risk_score"] >= 70,
        df["risk_score"] >= 40,
        df["risk_score"] >= 15,
    ]
    labels = ["HIGH","MEDIUM","LOW"]
    df["risk_level"] = np.select(conditions, labels, default="NORMAL")
    return df

def _justification(df: pd.DataFrame) -> pd.DataFrame:
    def build_reason(row):
        reasons = []
        for col, rule in RULES.items():
            if col not in row.index:
                continue
            if col in ["speed_anomaly", "altitude_anomaly", "vertical_anomaly"]:
                if row[col] == 1:
                    reasons.append(rule["label"])

            elif col == "altitude_diff":
                if pd.notna(row[col])and abs(row[col]) > 100:
                    reasons.append(rule["label"])

            elif col == "vertical_rate_abs":
                if pd.notna(row[col]) and row[col] >10:
                    reasons.append(rule["label"])

        return " | ".join(reasons) if reasons else "Aucune anomalie détectée"
    df["justification"] = df.apply(build_reason, axis=1)
    return df

def save_scores(df: pd.DataFrame) -> str:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    path = "data/processed/scored.csv"
    df.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    df = compute_risk(df)
    path = save_scores(df)
    print(f"Scored: {len(df)} Vols")
    print(df["risk_level"].value_counts())
    print(f"Saved -> {path}")
