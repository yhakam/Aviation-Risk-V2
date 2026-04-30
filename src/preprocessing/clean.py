import pandas as pd
import numpy as np
from pathlib import Path

REQUIRED_COLUMNS = ["icao24", "latitude", "longitude", "velocity", "baro_altitude"]
NULL_THRESHOLD = 0.4 #le seuil de valeur nulle on le met a 40% pour garder seulement les lignes pertinentes

def clean(df: pd.DataFrame) -> pd.DataFrame: #on néttoie les données brutes
    df = df.copy() # On crée une copie pour pas modifier le DataFrame originale
    df = _cast_types(df) # On applique le bon types aux données numériques
    df = _convert_timestamps(df) # On convertit les colonnes de temps en date
    df = _drop_unusable_rows(df) # On supprime les colonnes inutilisables
    return df

def _drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["on_ground"]] #le risque est défini uniquement pour les avions en vol
    null_ratio = df.isnull().sum(axis=1) / len(df.columns) # Valeur nulles présentes dans une ligne / le nombre de colonnes
    df = df[null_ratio < NULL_THRESHOLD]

    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df

def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ["latitude","longitude","velocity","baro_altitude","geo_altitude","vertical_rate","true_track"]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") #si jamais une valeur peut pas être convertible on la transforme en NaN
    if "on_ground" in df.columns:        
        df["on_ground"] = df["on_ground"].astype(bool)
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].str.strip()
    return df

def _convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df["time_position"] = pd.to_datetime(df["time_position"], unit="s", utc=True)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], unit="s", utc=True)
    return df

def save_clean(df: pd.DataFrame) -> str:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    path = "data/processed/clean.csv"
    df.to_csv(path, index=False)
    return path   

if __name__ == "__main__":
    raw_path= sorted(Path("data/raw").glob("*.csv"))[-1]
    df_raw = pd.read_csv(raw_path)
    df_clean = clean(df_raw)
    path = save_clean(df_clean)
    print(f"Cleaned: {len(df_clean)} rows saved to {path}")