"""
analyze.py
----------
Module d'analyse : visualisations du parcours client multitouch.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from pathlib import Path
from src.load import query_sqlite


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


def plot_attribution_comparaison() -> None:
    """
    Comparaison des 4 modèles d'attribution par canal.
    C'est le graphique star du projet.
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
    ax.set_title("Comparaison des modèles d'attribution par canal",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Modèle d'attribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "attribution_comparaison.png", dpi=150)
    plt.close()
    print("✅ attribution_comparaison.png sauvegardé")


def plot_taux_conversion_canal() -> None:
    """Taux de conversion last-touch par canal."""
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
    print("✅ taux_conversion_canal.png sauvegardé")


def plot_parcours_longueur() -> None:
    """Distribution du nombre de touchpoints avant conversion vs non-conversion."""
    df_conv = query_sqlite("""
        SELECT n_touches_total, converti
        FROM touchpoints
        WHERE is_last_touch = 1
    """)

    fig, ax = plt.subplots(figsize=(10, 5))

    for converti, label, couleur in [
        (1, "Converti",     "#16A34A"),
        (0, "Non converti", "#DC2626"),
    ]:
        subset = df_conv[df_conv["converti"] == converti]["n_touches_total"]
        ax.hist(subset, bins=range(1, 8), alpha=0.6, label=label,
                color=couleur, edgecolor="white", density=True)

    ax.set_xlabel("Nombre de touchpoints")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des touchpoints : convertis vs non-convertis",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(1, 7))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "distribution_touchpoints.png", dpi=150)
    plt.close()
    print("✅ distribution_touchpoints.png sauvegardé")


def plot_canal_position() -> None:
    """
    Heatmap : fréquence d'apparition de chaque canal
    selon sa position dans le parcours (1er, 2ème, etc.).
    """
    df = query_sqlite("""
        SELECT canal, position, COUNT(*) as nb
        FROM touchpoints
        WHERE position <= 5
        GROUP BY canal, position
        ORDER BY canal, position
    """)

    pivot = df.pivot(index="canal", columns="position", values="nb").fillna(0)
    pivot.columns = [f"Position {c}" for c in pivot.columns]

    # Normalisation par ligne (en % de chaque canal)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
    ax.set_title("Position des canaux dans le parcours client (%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Position dans le parcours")
    ax.set_ylabel("Canal")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "heatmap_position_canal.png", dpi=150)
    plt.close()
    print("✅ heatmap_position_canal.png sauvegardé")


def plot_conversion_par_segment() -> None:
    """Taux de conversion par segment client et par canal."""
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
    print("✅ conversion_par_segment.png sauvegardé")


def run_analysis() -> None:
    """Lance toutes les visualisations."""
    print("=== ANALYSE & VISUALISATION ===")
    plot_attribution_comparaison()
    plot_taux_conversion_canal()
    plot_parcours_longueur()
    plot_canal_position()
    plot_conversion_par_segment()
    print("\n✅ Toutes les visualisations sont dans outputs/")