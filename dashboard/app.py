import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title ="Aviation Risk Dashboard",layout="wide")

RISK_COLORS = {"HIGH": "#E24B4A", "MEDIUM": "#EF9F27", "LOW": "#378ADD", "NORMAL":"#888780"}

@st.cache_data

def load_data() -> pd.DataFrame:
    path = Path("data/processed/benchmark.csv")
    if not path.exists():
        path = Path("data/sample/benchmark_sample.csv")

        if not path.exists():
            st.error("Aucune donnée disponible. Ajoutez un sample")
            st.stop()
    df = pd.read_csv(path)
    df["callsign"] = df["callsign"].fillna("UNKNOWN").str.strip()
    return df

df = load_data()

st.sidebar.title("Filtres")

risk_filter = st.sidebar.multiselect("Niveau de Risque",options = ["HIGH", "MEDIUM", "LOW", "NORMAL"], default = ["HIGH","MEDIUM","LOW"])

score_min, score_max = st.sidebar.slider("Risk Score", min_value=0.0, max_value=100.0, value=(0.0,100.0), step=0.5)

country_options = sorted(df["origin_country"].dropna().unique())
country_filter = st.sidebar.multiselect("Pays d'Origine", options=country_options, default=[])

df_filtered = df[df["risk_level"].isin(risk_filter)]
df_filtered = df_filtered[(df_filtered["risk_score"] >= score_min) & (df_filtered["risk_score"] <= score_max)]

if country_filter:
    df_filtered = df_filtered[df_filtered["origin_country"].isin(country_filter)]

if df_filtered.empty: 
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

st.title("Aviation Risk Dashboard")
st.markdown("""
            
            Ce dashboard combine à la fois un score de risque basé sur des règles métier (vitesse, altitude, dynamique du vol) avec un modèle d'anomalie non supervisé (Isolation Forest).
            
            L'objectif est d'identifier des situations de vols potentiellement anormales tout en conservant une approche explicables et compréhensible""")
st.caption(f"{len(df_filtered):,} vols affichés sur {len(df):,} total")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("HIGH", df_filtered[df_filtered["risk_level"] == "HIGH"].shape[0])
col2.metric("MEDIUM", df_filtered[df_filtered["risk_level"] == "MEDIUM"].shape[0])
col3.metric("LOW", df_filtered[df_filtered["risk_level"] == "LOW"].shape[0])
avg_score = df_filtered["risk_score"].mean()
col4.metric("Score moyen", f"{avg_score:.1f}")
col5.metric("Anomalies ML", df_filtered["anomaly_flag"].sum())

high_count = df_filtered[df_filtered["risk_level"] == "HIGH"].shape[0]

if high_count == 0:
    st.info("Aucun vol classé HIGH n'a été détecté dans cet échantillon, ce qui est cohérent avec un trafic commercial majoritairement stable. " 
            "Les vols MEDIUM correspondent à des situations à surveiller, pas nécessairement dangereuses.")

st.subheader("Carte des vols")

fig_map = px.scatter_map(df_filtered, lat = "latitude", lon = "longitude", color = "risk_level", color_discrete_map=RISK_COLORS,hover_name = "callsign",
                         hover_data= {"risk_score": True, "risk_level": True, "justification": True, "origin_country": True, "latitude": False, "longitude": False}, zoom=2, height= 550)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, width = "stretch")

st.subheader("Analyse")

col_left, col_right = st.columns(2)

with col_left:
    fig_dist = px.histogram(df_filtered, x="risk_score", color="risk_level", color_discrete_map = RISK_COLORS, nbins=40, title = "Distribution des score de risque", labels = {"risk_score": "Score de risque", "count": "Nombre de vols"})
    st.plotly_chart(fig_dist, width = "stretch")

with col_right:
    MIN_FLIGHTS = 10
    top_countries = (df_filtered.groupby("origin_country").agg(risk_score=("risk_score", "mean"),flight_count=("risk_score", "count")).reset_index())

    top_countries = (top_countries[top_countries["flight_count"] >= MIN_FLIGHTS].sort_values(by="risk_score", ascending=False).head(15))

    fig_countries = px.bar(top_countries,x="risk_score",y="origin_country",orientation="h",text="flight_count",title=f"Score de risque moyen par pays (top 15, min {MIN_FLIGHTS} vols)",labels={"risk_score": "Score moyen","origin_country": "Pays","flight_count": "Nombre de vols"})

    fig_countries.update_traces(textposition="outside")

    st.plotly_chart(fig_countries, width="stretch")

    st.caption(
        f"Seuls les pays avec au moins {MIN_FLIGHTS} vols sont inclus afin d'éviter les moyennes peu représentatives."
    )

st.subheader("Rule-Based vs Isolation Forest")

fig_comp = px.scatter(df_filtered, x = "risk_score", y = "anomaly_score", color = "risk_level", color_discrete_map = RISK_COLORS, hover_name="callsign",
                      hover_data={"justification": True}, title = "Corrélation risk_score (règles) vs anomaly_score (ML)", labels={"risk_score": "Risk score (rule_based)", "anomaly_score": "Anomaly score (Isolation Forest)"},height = 450)

st.plotly_chart(fig_comp, width = "stretch")

st.subheader("Vols les plus risqué")

cols_display = ["callsign", "origin_country", "risk_score", "risk_level", "anomaly_score", "justification"]

df_display = (df_filtered[df_filtered["risk_level"].isin(["HIGH","MEDIUM"])][cols_display].sort_values("risk_score", ascending=False).reset_index(drop=True))

if df_display.empty:
    st.info("Aucun vol HIGH ou MEDIUM avec les filtres actuels.")
else:
    st.dataframe(df_display, width = "stretch")