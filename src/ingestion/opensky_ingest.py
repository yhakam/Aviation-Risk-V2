import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

URL = "https://opensky-network.org/api/states/all"

def fetch_data():
    try:
        response = requests.get(URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Erreur API: {e}") from e 

def to_dataframe(data):
    columns = ["icao24","callsign","origin_country", "time_position", "last_contact","longitude", "latitude", "baro_altitude",
        "on_ground", "velocity", "true_track", "vertical_rate",
        "sensors", "geo_altitude", "squawk", "spi", "position_source"]
    states = data.get("states") or []
    df = pd.DataFrame(states,columns=columns)
    df["snapshot_time"] = data.get("time")

    return df

def save(df):
    Path("data/raw").mkdir(parents=True,exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"data/raw/opensky_{timestamp}.csv"
    df.to_csv(path,index=False)
    return path

def main():
    data = fetch_data()
    df = to_dataframe(data)
    path = save(df)

    print(f"Saved:{path}")

if __name__ == "__main__":
    main()