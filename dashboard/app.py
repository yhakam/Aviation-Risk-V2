import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Tableau de bord des risques en aviation", layout="wide")

RISK_LABELS = {"HIGH": "ÉLEVÉ", "MEDIUM": "MOYEN", "LOW": "FAIBLE", "NORMAL": "NORMAL"}
RISK_COLORS = {"ÉLEVÉ": "#E24B4A", "MOYEN": "#EF9F27", "FAIBLE": "#378ADD", "NORMAL": "#888780"}

@st.cache_data
def load_data() -> pd.DataFrame:
    path = Path("data/processed/benchmark.csv")
    if not path.exists():
        path = Path("data/sample/benchmark_sample.csv")

        if not path.exists():
            st.error("Aucune donnée disponible. Ajoutez un échantillon.")
            st.stop()

    df = pd.read_csv(path)
    df["callsign"] = df["callsign"].fillna("INCONNU").str.strip()
    df["risk_level_fr"] = df["risk_level"].map(RISK_LABELS).fillna(df["risk_level"])
    return df

df = load_data()

st.sidebar.title("Filtres")

risk_filter = st.sidebar.multiselect("Niveau de risque", options=["ÉLEVÉ", "MOYEN", "FAIBLE", "NORMAL"], default=["ÉLEVÉ", "MOYEN", "FAIBLE"])

score_min, score_max = st.sidebar.slider("Score de risque", min_value=0.0, max_value=100.0, value=(0.0, 100.0), step=0.5)

country_options = sorted(df["origin_country"].dropna().unique())
country_filter = st.sidebar.multiselect("Pays d'origine", options=country_options, default=[])

df_filtered = df[df["risk_level_fr"].isin(risk_filter)]
df_filtered = df_filtered[(df_filtered["risk_score"] >= score_min) & (df_filtered["risk_score"] <= score_max)]

if country_filter:
    df_filtered = df_filtered[df_filtered["origin_country"].isin(country_filter)]

if df_filtered.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

st.title("Tableau de bord des risques en aviation")
st.markdown("""
Ce tableau de bord combine un score de risque basé sur des règles métier (vitesse, altitude, dynamique du vol)
avec un modèle d'anomalie non supervisé (Isolation Forest).

L'objectif est d'identifier des situations de vol potentiellement anormales tout en conservant une approche explicable et compréhensible.
""")

st.caption(f"{len(df_filtered):,} vols affichés sur {len(df):,} au total")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ÉLEVÉ", df_filtered[df_filtered["risk_level"] == "HIGH"].shape[0])
col2.metric("MOYEN", df_filtered[df_filtered["risk_level"] == "MEDIUM"].shape[0])
col3.metric("FAIBLE", df_filtered[df_filtered["risk_level"] == "LOW"].shape[0])
avg_score = df_filtered["risk_score"].mean()
col4.metric("Score moyen", f"{avg_score:.1f}")
col5.metric("Anomalies ML", df_filtered["anomaly_flag"].sum())

high_count = df_filtered[df_filtered["risk_level"] == "HIGH"].shape[0]

if high_count == 0:
    st.info(
        "Aucun vol classé ÉLEVÉ n'a été détecté dans cet échantillon, ce qui est cohérent avec un trafic commercial majoritairement stable. "
        "Les vols MOYENS correspondent à des situations à surveiller, pas nécessairement dangereuses."
    )

st.subheader("Carte des vols")

fig_map = px.scatter_map(df_filtered,lat="latitude",lon="longitude",color="risk_level_fr",color_discrete_map=RISK_COLORS,hover_name="callsign",
    hover_data={"risk_score": True, "risk_level_fr": True, "justification": True, "origin_country": True, "latitude": False, "longitude": False},zoom=2,height=550)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, width="stretch")

st.subheader("Analyse")

col_left, col_right = st.columns(2)

with col_left:
    fig_dist = px.histogram(df_filtered,x="risk_score",color="risk_level_fr",color_discrete_map=RISK_COLORS,nbins=40,title="Distribution des scores de risque",labels={"risk_score": "Score de risque", "count": "Nombre de vols", "risk_level_fr": "Niveau de risque"})
    st.plotly_chart(fig_dist, width="stretch")

with col_right:
    MIN_FLIGHTS = 10
    top_countries = (df_filtered.groupby("origin_country").agg(risk_score=("risk_score", "mean"), flight_count=("risk_score", "count")).reset_index())

    top_countries = (top_countries[top_countries["flight_count"] >= MIN_FLIGHTS].sort_values(by="risk_score", ascending=False).head(15))
    if top_countries.empty:
        st.info(f"Aucun pays ne possède au moins {MIN_FLIGHTS} vols avec les filtres actuels")
    else:
        fig_countries = px.bar(top_countries,x="risk_score",y="origin_country",color="risk_score",color_continuous_scale=["#378ADD", "#EF9F27", "#E24B4A"],orientation="h",text="flight_count",title=f"Score de risque moyen par pays (top 15, min {MIN_FLIGHTS} vols)",labels={"risk_score": "Score moyen", "origin_country": "Pays", "flight_count": "Nombre de vols"})

        fig_countries.update_traces(textposition="outside")
    
        fig_countries.update_layout(yaxis=dict(autorange="reversed"))

        st.plotly_chart(fig_countries, width="stretch")

    st.caption(f"Seuls les pays avec au moins {MIN_FLIGHTS} vols sont inclus afin d'éviter les moyennes peu représentatives.")

st.subheader("Approche métier vs Isolation Forest")

fig_comp = px.scatter(df_filtered,x="risk_score",y="anomaly_score",color="risk_level_fr",color_discrete_map=RISK_COLORS,hover_name="callsign",hover_data={"justification": True},
                      title="Corrélation entre le score de risque métier et le score d'anomalie ML",labels={"risk_score": "Score de risque métier", "anomaly_score": "Score d'anomalie ML", "risk_level_fr": "Niveau de risque"},height=450)

st.plotly_chart(fig_comp, width="stretch")

st.subheader("Vols les plus risqués")

cols_display = ["callsign", "origin_country", "risk_score", "risk_level_fr", "anomaly_score", "justification"]

df_display = (df_filtered[df_filtered["risk_level"].isin(["HIGH", "MEDIUM"])][cols_display].sort_values("risk_score", ascending=False).reset_index(drop=True))
df_display = df_display.rename(columns={"callsign": "Vol","origin_country": "Pays","risk_score": "Score de risque","risk_level_fr": "Niveau","anomaly_score": "Score ML","justification": "Justification",})

if df_display.empty:
    st.info("Aucun vol ÉLEVÉ ou MOYEN avec les filtres actuels.")
else:
    st.dataframe(df_display, width="stretch")