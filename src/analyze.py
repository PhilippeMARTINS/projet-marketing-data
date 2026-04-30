"""
analyze.py
----------
Module d'analyse : génération des 5 visualisations marketing
(attribution, conversion canal, longueur parcours, position canal, conversion segment).
Les graphiques sont sauvegardés dans outputs/ au format PNG.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from src.load import query_sqlite


# ── Configuration ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

COULEURS_CANAUX = {
    "Email":      "#2563EB",
    "Google Ads": "#16A34A",
    "SEO":        "#D97706",
    "Instagram":  "#7C3AED",
    "Facebook":   "#DC2626",
    "YouTube":    "#0891B2",
}


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_attribution_comparaison() -> None:
    """
    Comparaison des 4 modèles d'attribution par canal.
    Colonnes utilisées : last_click, first_click, linear, time_decay.
    """
    df = query_sqlite("SELECT * FROM attribution ORDER BY last_click DESC")

    modeles = ["last_click", "first_click", "linear", "time_decay"]
    labels  = ["Last Click", "First Click", "Linear", "Time Decay"]
    canaux  = df["canal"].tolist()
    x       = np.arange(len(canaux))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (modele, label) in enumerate(zip(modeles, labels)):
        ax.bar(x + i * width, df[modele], width, label=label, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(canaux, fontsize=11)
    ax.set_ylabel("Part d'attribution (%)")
    ax.set_title("Comparaison des modeles d'attribution par canal",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Modele d'attribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "attribution_comparaison.png", dpi=150)
    plt.close()
    logger.info("Visualisation sauvegardee : attribution_comparaison.png")


def plot_taux_conversion_canal() -> None:
    """Taux de conversion last-touch par canal (en %)."""
    df = query_sqlite("""
        SELECT canal, taux_conversion, nb_conversions
        FROM canal_stats
        ORDER BY taux_conversion DESC
    """)

    couleurs = [COULEURS_CANAUX.get(c, "#6B7280") for c in df["canal"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["canal"][::-1], df["taux_conversion"][::-1],
                   color=couleurs[::-1], alpha=0.85)

    for bar, val in zip(bars, df["taux_conversion"][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Taux de conversion (%)")
    ax.set_title("Taux de conversion last-touch par canal",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, df["taux_conversion"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "taux_conversion_canal.png", dpi=150)
    plt.close()
    logger.info("Visualisation sauvegardee : taux_conversion_canal.png")


def plot_parcours_longueur() -> None:
    """Distribution du nombre de touchpoints par client."""
    df = query_sqlite("""
        SELECT client_id, MAX(n_touches_total) AS n_touches
        FROM touchpoints
        GROUP BY client_id
    """)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["n_touches"], bins=6, kde=False, color="#2563EB", ax=ax)
    ax.set_xlabel("Nombre de touchpoints")
    ax.set_ylabel("Nombre de clients")
    ax.set_title("Distribution de la longueur du parcours client",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "distribution_touchpoints.png", dpi=150)
    plt.close()
    logger.info("Visualisation sauvegardee : distribution_touchpoints.png")


def plot_canal_position() -> None:
    """Heatmap du taux de conversion par canal et position dans le parcours."""
    df = query_sqlite("""
        SELECT canal,
               position,
               ROUND(AVG(converti) * 100, 2) AS taux_conversion
        FROM touchpoints
        WHERE position <= 5
        GROUP BY canal, position
        ORDER BY canal, position
    """)

    pivot = df.pivot(index="canal", columns="position",
                     values="taux_conversion").fillna(0)
    pivot.columns = [f"Position {c}" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Taux de conversion (%)"})
    ax.set_title("Taux de conversion par canal et position dans le parcours",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "heatmap_position_canal.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Visualisation sauvegardee : heatmap_position_canal.png")


def plot_conversion_par_segment() -> None:
    """Barres groupees du taux de conversion par canal et segment client."""
    df = query_sqlite("""
        SELECT t.canal, c.segment,
               ROUND(AVG(t.converti) * 100, 2) AS taux_conversion
        FROM touchpoints t
        JOIN clients c ON t.client_id = c.client_id
        WHERE t.is_last_touch = 1
        GROUP BY t.canal, c.segment
        ORDER BY t.canal, taux_conversion DESC
    """)

    pivot = df.pivot(index="canal", columns="segment",
                     values="taux_conversion").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, alpha=0.85, edgecolor="white")
    ax.set_title("Taux de conversion par canal et segment client",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Canal")
    ax.set_ylabel("Taux de conversion (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.xticks(rotation=15)
    ax.legend(title="Segment client", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "conversion_par_segment.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Visualisation sauvegardee : conversion_par_segment.png")


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_analysis() -> None:
    """
    Lance la generation de toutes les visualisations et les sauvegarde dans outputs/.

    Visualisations produites :
        - attribution_comparaison.png
        - taux_conversion_canal.png
        - distribution_touchpoints.png
        - heatmap_position_canal.png
        - conversion_par_segment.png
    """
    logger.info("Generation des visualisations...")
    plot_attribution_comparaison()
    plot_taux_conversion_canal()
    plot_parcours_longueur()
    plot_canal_position()
    plot_conversion_par_segment()
    logger.info("Toutes les visualisations sont dans '%s'", OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_analysis()