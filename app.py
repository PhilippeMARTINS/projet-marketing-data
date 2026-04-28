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
st.caption("Pipeline ETL · Python · Pandas · SQLite · LightGBM · Streamlit")
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
    bars_attr = ax1.bar(x + i * width, df_attr[modele], width,
                        label=label, alpha=alpha)
    # Affiche le pourcentage au-dessus uniquement pour le modèle mis en avant
    if modele == modele_choisi:
        for bar in bars_attr:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.3,
                f"{height:.1f}%",
                ha="center", va="bottom",
                fontsize=15, fontweight="bold",
                color="#1E3A5F"  # bleu foncé, visible sur toutes les couleurs de barres
            )

ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(df_attr["canal"])
ax1.tick_params(axis='x', labelsize=17)
ax1.tick_params(axis='y', labelsize=17)
ax1.set_ylabel("Part d'attribution", fontsize=17)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax1.legend(title="Modèle")
plt.tight_layout()
st.pyplot(fig1)
plt.close()

st.markdown("---")


# ── Graphique 2 — Taux de conversion par canal ────────────────────────────────
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
             f"{val:.1f}%", va="center", fontweight="bold")
ax2.set_xlabel("Taux de conversion (%)", fontsize=14)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
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
for container in ax3.containers:
    ax3.bar_label(container, fmt="%.0f%%", padding=3, fontsize=11)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.set_xlabel("Canal",fontsize=17)
ax3.set_ylabel("Taux de conversion", fontsize=17)
ax3.tick_params(axis='x', labelsize=15)
ax3.tick_params(axis='y', labelsize=15)
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


# ── Graphique 5 — CPA par canal ───────────────────────────────────────────────
st.subheader("💰 Coût par acquisition (CPA) par canal")
st.caption("Coût moyen pour générer une conversion — hors SEO (canal organique, CPA = 0€)")

df_roi = query(f"""
    SELECT canal, cpa, cout_total, nb_conversions, roi, type_facturation
    FROM roi_by_canal
    WHERE canal IN ({canaux_sql})
      AND cpa > 0
    ORDER BY cpa ASC
""")

couleurs_cpa = [COULEURS_CANAUX.get(c, "#6B7280") for c in df_roi["canal"]]

fig5, ax5 = plt.subplots(figsize=(10, 4))
bars5 = ax5.barh(df_roi["canal"], df_roi["cpa"],
                 color=couleurs_cpa, alpha=0.85)
for bar, val in zip(bars5, df_roi["cpa"]):
    ax5.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}€", va="center")
ax5.set_xlabel("CPA (€)")
ax5.set_xlim(0, df_roi["cpa"].max() * 1.20)
ax5.set_title("Plus le CPA est bas, plus le canal est rentable",
              color="#6B7280", pad=8)
plt.tight_layout()
st.pyplot(fig5)
plt.close()

st.markdown("---")


# ── Graphique 6 — ROI par canal ───────────────────────────────────────────────
st.subheader("📈 ROI par canal marketing")
st.caption("Retour sur investissement = (Valeur générée - Coût) / Coût × 100 — hors SEO (organique)")

df_roi_chart = df_roi.sort_values("roi", ascending=True)
couleurs_roi = [COULEURS_CANAUX.get(c, "#6B7280") for c in df_roi_chart["canal"]]

fig6, ax6 = plt.subplots(figsize=(10, 4))
bars6 = ax6.barh(df_roi_chart["canal"], df_roi_chart["roi"],
                 color=couleurs_roi, alpha=0.85)
for bar, val in zip(bars6, df_roi_chart["roi"]):
    ax6.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
             f"{val:,.0f}%", va="center" )
ax6.set_xlabel("ROI")
ax6.set_xlim(0, df_roi_chart["roi"].max() * 1.15)
ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))
ax6.set_title("Valeur générée vs coût investi par canal",
              color="#6B7280", pad=8)
plt.tight_layout()
st.pyplot(fig6)
plt.close()

st.markdown("---")


# ── Graphique 7 — Bulle : Volume × Taux conversion × CPA ─────────────────────
st.subheader("🫧 Vue d'ensemble : Volume × Taux de conversion × CPA")
st.caption("Taille des bulles = nombre de conversions · Idéal : en haut à gauche (fort taux, faible CPA)")

df_bubble = query(f"""
    SELECT
        cs.canal,
        cs.taux_conversion,
        cs.nb_conversions,
        r.cpa
    FROM canal_stats cs
    LEFT JOIN roi_by_canal r ON cs.canal = r.canal
    WHERE cs.canal IN ({canaux_sql})
""")

fig7, ax7 = plt.subplots(figsize=(12, 6))
for _, row in df_bubble.iterrows():
    canal = row["canal"]
    color = COULEURS_CANAUX.get(canal, "#6B7280")
    cpa   = row["cpa"] if pd.notna(row["cpa"]) and row["cpa"] > 0 else 0
    size  = row["nb_conversions"] / 10

    ax7.scatter(
        cpa,
        row["taux_conversion"],
        s=size,
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidth=1.5,
    )
    ax7.annotate(
        f"{canal}\n{int(row['nb_conversions']):,} conv.",
        xy=(cpa, row["taux_conversion"]),
        xytext=(8, 4),
        textcoords="offset points",
        fontweight="bold",
        color=color,
    )

ax7.set_xlabel("CPA — Coût par acquisition", fontsize=15)
ax7.set_ylabel("Taux de conversion last-touch", fontsize=15)
ax7.tick_params(axis='x', labelsize=15)
ax7.tick_params(axis='y', labelsize=15)
ax7.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax7.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}€"))
plt.tight_layout()
st.pyplot(fig7)
plt.close()

st.markdown("---")


# ── Section Bilan & Recommandations ───────────────────────────────────────────
st.subheader("📌 Bilan & Recommandations")
st.caption("Insights générés automatiquement depuis les données")

# Chargement des données pour le bilan
df_bilan_stats = query(f"""
    SELECT canal, taux_conversion, nb_conversions
    FROM canal_stats
    WHERE canal IN ({canaux_sql})
    ORDER BY taux_conversion DESC
""")
df_bilan_roi = query(f"""
    SELECT canal, cpa, roi, cout_total, valeur_generee
    FROM roi_by_canal
    WHERE canal IN ({canaux_sql})
""")
df_bilan_seg = query(f"""
    SELECT c.segment,
           ROUND(AVG(t.converti) * 100, 2) AS taux_conversion
    FROM touchpoints t
    JOIN clients c ON t.client_id = c.client_id
    WHERE t.is_last_touch = 1
      AND c.segment IN ({segs_sql})
    GROUP BY c.segment
    ORDER BY taux_conversion DESC
""")
df_bilan_touches = query("""
    SELECT n_touches_total,
           ROUND(AVG(converti) * 100, 2) AS taux_conversion
    FROM touchpoints
    WHERE is_last_touch = 1
    GROUP BY n_touches_total
    ORDER BY n_touches_total
""")

# Calcul des insights dynamiques
meilleur_canal    = df_bilan_stats.iloc[0]["canal"]
meilleur_taux     = df_bilan_stats.iloc[0]["taux_conversion"]
moins_bon_canal   = df_bilan_stats.iloc[-1]["canal"]
moins_bon_taux    = df_bilan_stats.iloc[-1]["taux_conversion"]
meilleur_segment  = df_bilan_seg.iloc[0]["segment"]
moins_bon_segment = df_bilan_seg.iloc[-1]["segment"]
ratio_segments    = (df_bilan_seg.iloc[0]["taux_conversion"] /
                     df_bilan_seg.iloc[-1]["taux_conversion"])

df_roi_payants    = df_bilan_roi[df_bilan_roi["cpa"] > 0].sort_values("cpa")
meilleur_cpa_canal = df_roi_payants.iloc[0]["canal"] if not df_roi_payants.empty else "N/A"
meilleur_cpa       = df_roi_payants.iloc[0]["cpa"] if not df_roi_payants.empty else 0
meilleur_roi_canal = (df_roi_payants.sort_values("roi", ascending=False).iloc[0]["canal"]
                      if not df_roi_payants.empty else "N/A")
meilleur_roi       = (df_roi_payants.sort_values("roi", ascending=False).iloc[0]["roi"]
                      if not df_roi_payants.empty else 0)

taux_1_touch = df_bilan_touches[df_bilan_touches["n_touches_total"] == 1]["taux_conversion"].values
taux_4_touch = df_bilan_touches[df_bilan_touches["n_touches_total"] >= 4]["taux_conversion"].mean()
taux_1_touch = taux_1_touch[0] if len(taux_1_touch) > 0 else 0

valeur_totale = df_bilan_roi["valeur_generee"].sum()
cout_total    = df_bilan_roi["cout_total"].sum()

# Affichage du bilan en deux colonnes
col_b1, col_b2 = st.columns(2)

with col_b1:
    st.markdown("#### 🔍 Insights clés")
    st.markdown(f"""
- 📧 **{meilleur_canal}** est le canal le plus performant en last-touch avec **{meilleur_taux:.1f}%** de taux de conversion
- 📉 **{moins_bon_canal}** est le moins performant avec **{moins_bon_taux:.1f}%** de taux de conversion
- 👑 Les clients **{meilleur_segment}** convertissent **{ratio_segments:.1f}x** mieux que les **{moins_bon_segment}**
- 🔗 Un parcours de **4+ touchpoints** génère **{taux_4_touch:.1f}%** de conversion vs **{taux_1_touch:.1f}%** pour un seul touch
- 💰 **{meilleur_cpa_canal}** est le canal payant le plus rentable avec un CPA de **{meilleur_cpa:.2f}€**
- 📈 **{meilleur_roi_canal}** affiche le meilleur ROI parmi les canaux payants : **{meilleur_roi:,.0f}%**
    """)

with col_b2:
    st.markdown("#### 💡 Recommandations")
    st.markdown(f"""
- 🎯 **Maximiser Email** en closing : fort taux de conversion et CPA quasi nul ({meilleur_cpa:.2f}€)
- 🔍 **Investir en SEO** : canal organique (CPA = 0€) à fort potentiel de volume
- 🛒 **Cibler les {meilleur_segment}** en priorité : conversion {ratio_segments:.1f}x supérieure à la moyenne
- 🔗 **Favoriser les parcours multitouch** : 4+ touchpoints améliorent significativement la conversion
- ⚠️ **Réévaluer Facebook** : CPA le plus élevé des canaux payants, ROI le plus faible
- 📊 **Utiliser Google Ads** pour la phase de découverte : bon volume et ROI solide
    """)

st.markdown("---")

# KPIs financiers globaux
st.markdown("#### 📊 Synthèse financière")
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.metric("💸 Coût marketing total",
              f"{cout_total:,.0f}€".replace(",", " "))
col_f2.metric("💎 Valeur générée totale",
              f"{valeur_totale:,.0f}€".replace(",", " "))
col_f3.metric("📈 ROI global",
              f"{((valeur_totale - cout_total) / cout_total * 100):,.0f}%".replace(",", " ")
              if cout_total > 0 else "N/A")

st.markdown("---")


# ── Section SQL ────────────────────────────────────────────────────────────────
st.subheader("🧮 Requête SQL personnalisée")
st.caption("Tables disponibles : `clients`, `touchpoints`, `attribution`, `canal_stats`, `roi_by_canal`, `canal_costs`, `conversion_value`")

default_sql = """SELECT canal, cpa, roi, valeur_generee
FROM roi_by_canal
ORDER BY roi DESC"""

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