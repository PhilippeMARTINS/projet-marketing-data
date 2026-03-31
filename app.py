"""
app.py
------
Dashboard Streamlit — Analyse du parcours client multitouch.
Lancer avec : streamlit run app.py
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import streamlit as st
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH = Path("data/processed/marketing.db")
sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Marketing Dashboard — Parcours Client",
    page_icon="📊",
    layout="wide",
)

COULEURS_CANAUX = {
    "Email":      "#2563EB",
    "Google Ads": "#16A34A",
    "SEO":        "#D97706",
    "Instagram":  "#7C3AED",
    "Facebook":   "#DC2626",
    "YouTube":    "#0891B2",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Filtres")

segments = query("SELECT DISTINCT segment FROM clients ORDER BY segment")["segment"].tolist()
selected_segments = st.sidebar.multiselect(
    "Segment client", options=segments, default=segments
)

canaux = query("SELECT DISTINCT canal FROM canal_stats ORDER BY canal")["canal"].tolist()
selected_canaux = st.sidebar.multiselect(
    "Canal", options=canaux, default=canaux
)

if not selected_segments:
    selected_segments = segments
if not selected_canaux:
    selected_canaux = canaux

segs_sql   = ", ".join(f"'{s}'" for s in selected_segments)
canaux_sql = ", ".join(f"'{c}'" for c in selected_canaux)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Astuce** : laisse vide pour tout afficher.")


# ── Titre ──────────────────────────────────────────────────────────────────────
st.title("📊 Marketing Dashboard — Parcours Client Multitouch")
st.caption("Pipeline ETL · Python · Pandas · SQLite · Scikit-learn · Streamlit")
st.markdown("---")


# ── KPIs globaux ───────────────────────────────────────────────────────────────
kpi_sql = f"""
    SELECT
        COUNT(DISTINCT t.client_id)              AS nb_clients,
        COUNT(*)                                  AS nb_touchpoints,
        ROUND(AVG(t.n_touches_total), 2)          AS touches_moyens,
        ROUND(SUM(t.converti) * 100.0 /
              COUNT(DISTINCT t.client_id), 2)     AS taux_conversion
    FROM touchpoints t
    JOIN clients c ON t.client_id = c.client_id
    WHERE t.is_last_touch = 1
      AND c.segment IN ({segs_sql})
      AND t.canal IN ({canaux_sql})
"""
kpi = query(kpi_sql).iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Clients",         f"{int(kpi['nb_clients']):,}".replace(",", " "))
col2.metric("🖱️ Touchpoints",     f"{int(kpi['nb_touchpoints']):,}".replace(",", " "))
col3.metric("📍 Touches moyens",  f"{kpi['touches_moyens']}")
col4.metric("🎯 Taux conversion", f"{kpi['taux_conversion']}%")

st.markdown("---")


# ── Graphique 1 — Attribution ──────────────────────────────────────────────────
st.subheader("🏆 Modèles d'attribution par canal")
st.caption("Comparaison Last Click / First Click / Linear / Time Decay")

df_attr = query(f"""
    SELECT * FROM attribution
    WHERE canal IN ({canaux_sql})
    ORDER BY last_click DESC
""")

modele_choisi = st.radio(
    "Modèle d'attribution à mettre en avant",
    ["last_click", "first_click", "linear", "time_decay"],
    horizontal=True,
    format_func=lambda x: {
        "last_click":  "Last Click",
        "first_click": "First Click",
        "linear":      "Linear",
        "time_decay":  "Time Decay",
    }[x],
)

modeles = ["last_click", "first_click", "linear", "time_decay"]
labels  = ["Last Click", "First Click", "Linear", "Time Decay"]
x       = np.arange(len(df_attr))
width   = 0.2

fig1, ax1 = plt.subplots(figsize=(14, 5))
for i, (modele, label) in enumerate(zip(modeles, labels)):
    alpha = 1.0 if modele == modele_choisi else 0.35
    ax1.bar(x + i * width, df_attr[modele], width,
            label=label, alpha=alpha)

ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(df_attr["canal"], fontsize=11)
ax1.set_ylabel("Part d'attribution (%)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax1.legend(title="Modèle")
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.markdown("---")


# ── Graphique 2 — Taux de conversion par canal ─────────────────────────────────
st.subheader("🎯 Taux de conversion last-touch par canal")

df_stats = query(f"""
    SELECT canal, taux_conversion, nb_conversions, nb_last_touch
    FROM canal_stats
    WHERE canal IN ({canaux_sql})
    ORDER BY taux_conversion DESC
""")

couleurs = [COULEURS_CANAUX.get(c, "#6B7280") for c in df_stats["canal"]]

fig2, ax2 = plt.subplots(figsize=(10, 4))
bars = ax2.barh(df_stats["canal"][::-1], df_stats["taux_conversion"][::-1],
                color=couleurs[::-1], alpha=0.85)
for bar, val in zip(bars, df_stats["taux_conversion"][::-1]):
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("Taux de conversion (%)")
ax2.set_xlim(0, df_stats["taux_conversion"].max() * 1.15)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")


# ── Graphique 3 — Conversion par segment ──────────────────────────────────────
st.subheader("👥 Conversion par segment client et canal")

df_seg = query(f"""
    SELECT t.canal, c.segment,
           ROUND(AVG(t.converti) * 100, 2) AS taux_conversion
    FROM touchpoints t
    JOIN clients c ON t.client_id = c.client_id
    WHERE t.is_last_touch = 1
      AND c.segment IN ({segs_sql})
      AND t.canal IN ({canaux_sql})
    GROUP BY t.canal, c.segment
""")

pivot = df_seg.pivot(index="canal", columns="segment",
                     values="taux_conversion").fillna(0)

fig3, ax3 = plt.subplots(figsize=(12, 4))
pivot.plot(kind="bar", ax=ax3, alpha=0.85, edgecolor="white")
ax3.set_xlabel("Canal")
ax3.set_ylabel("Taux de conversion (%)")
ax3.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=15)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")


# ── Graphique 4 — Heatmap position canal ──────────────────────────────────────
st.subheader("🗺️ Position des canaux dans le parcours client")

df_pos = query(f"""
    SELECT canal, position, COUNT(*) as nb
    FROM touchpoints
    WHERE position <= 5
      AND canal IN ({canaux_sql})
    GROUP BY canal, position
""")

pivot_pos = df_pos.pivot(index="canal", columns="position",
                          values="nb").fillna(0)
pivot_pos.columns = [f"Position {c}" for c in pivot_pos.columns]
pivot_pct = pivot_pos.div(pivot_pos.sum(axis=1), axis=0) * 100

fig4, ax4 = plt.subplots(figsize=(10, 4))
sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, ax=ax4, cbar_kws={"label": "%"})
ax4.set_xlabel("Position dans le parcours")
ax4.set_ylabel("Canal")
plt.tight_layout()
st.pyplot(fig4)
plt.close()

st.markdown("---")


# ── Section SQL ────────────────────────────────────────────────────────────────
st.subheader("🧮 Requête SQL personnalisée")
st.caption("Tables disponibles : `clients`, `touchpoints`, `attribution`, `canal_stats`")

default_sql = """SELECT canal, taux_conversion, nb_conversions
FROM canal_stats
ORDER BY taux_conversion DESC"""

custom_sql = st.text_area("Requête SQL", value=default_sql, height=120)

if st.button("▶️ Exécuter"):
    try:
        df_custom = query(custom_sql)
        st.success(f"{len(df_custom)} ligne(s) retournée(s)")
        st.dataframe(df_custom, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur SQL : {e}")

st.markdown("---")
st.caption("Projet réalisé par **Philippe Morais Martins** · M2 Data Engineering · Paris Ynov Campus")