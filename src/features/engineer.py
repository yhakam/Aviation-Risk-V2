import pandas as pd
from pathlib import Path

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _speed_features(df)
    df = _altitude_features(df)
    df = _vertical_features(df)
    df = _position_features(df)
    return df

def _speed_features(df: pd.DataFrame) -> pd.DataFrame:
    df["speed_ms"] = df["velocity"]
    df["speed_kmh"] = df["velocity"] * 3.6
    df["speed_anomaly"] = ((df["speed_kmh"]< 100) | (df["speed_kmh"] > 1200)).astype(int)
    return df

def _altitude_features(df: pd.DataFrame) -> pd.DataFrame:
    if "geo_altitude" in df.columns:
        df["altitude_diff"] =  df["baro_altitude"] - df["geo_altitude"]
    df["altitude_anomaly"] = ((df["baro_altitude"] < 0 ) | (df["baro_altitude"]>13000)).astype(int)
    return df

def _vertical_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_climbing"] = (df["vertical_rate"] > 0).astype(int)
    df["is_descending"] = (df["vertical_rate"] < 0).astype(int)
    df["vertical_rate_abs"] = df["vertical_rate"].abs()
    df["vertical_anomaly"] = (df["vertical_rate_abs"] > 50).astype(int)
    return df

def _position_features(df : pd.DataFrame) -> pd.DataFrame:
    df["lat_bin"] = pd.cut(df["latitude"], bins=18, labels=False)
    df["lon_bin"] = pd.cut(df["longitude"], bins=36, labels=False)
    return df

def save_features(df: pd.DataFrame) -> str:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    path = "data/processed/features.csv"
    df.to_csv(path, index=False)
    return path

if  __name__ == "__main__":
    df = pd.read_csv("data/processed/clean.csv")
    df = build_features(df)
    path = save_features(df)
    print(f"Features: {df.shape[1]} colonnes, {len(df)} lignes → {path}")