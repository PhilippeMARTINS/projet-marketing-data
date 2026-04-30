"""
transform.py
------------
Module de transformation : nettoyage, enrichissement et calcul des 4 modèles
d'attribution marketing (Last Click, First Click, Linear, Time Decay).

Noms des colonnes d'attribution :
    - last_click  : 100% du crédit au dernier canal
    - first_click : 100% du crédit au premier canal
    - linear      : crédit équiréparti (en %)
    - time_decay  : crédit pondéré exponentiellement (en %)
"""

import logging

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Nettoyage ─────────────────────────────────────────────────────────────────

def clean_datasets(datasets: dict) -> dict:
    """
    Nettoie et type les datasets bruts.

    Opérations :
        - Conversion de la colonne 'date' en datetime
        - Conversion des colonnes booléennes (is_first_touch, is_last_touch)

    Args:
        datasets: dict {"clients": DataFrame, "touchpoints": DataFrame}

    Returns:
        dict: Datasets nettoyés avec les bons types
    """
    clients     = datasets["clients"].copy()
    touchpoints = datasets["touchpoints"].copy()

    touchpoints["date"]           = pd.to_datetime(touchpoints["date"])
    touchpoints["is_first_touch"] = touchpoints["is_first_touch"].astype(bool)
    touchpoints["is_last_touch"]  = touchpoints["is_last_touch"].astype(bool)

    logger.info(
        "Nettoyage terminé — clients : %d lignes, touchpoints : %d lignes",
        len(clients), len(touchpoints),
    )
    return {"clients": clients, "touchpoints": touchpoints}


# ── Modèles d'attribution ─────────────────────────────────────────────────────

def compute_attribution(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 4 modèles d'attribution marketing par canal, en pourcentage.

    Modèles :
        - last_click  : 100% au dernier canal
        - first_click : 100% au premier canal
        - linear      : équiréparti (%)
        - time_decay  : pondéré exponentiellement (%)

    Seuls les parcours avec au moins une conversion sont pris en compte.

    Args:
        touchpoints: DataFrame de touchpoints issu de clean_datasets()

    Returns:
        pd.DataFrame: Attribution par canal (colonnes : canal, last_click,
                      first_click, linear, time_decay) — valeurs en %
    """
    clients_convertis = touchpoints[touchpoints["converti"] == 1]["client_id"].unique()
    df_conv = touchpoints[touchpoints["client_id"].isin(clients_convertis)].copy()

    canaux_uniques = df_conv["canal"].unique()
    resultats = {
        canal: {"last_click": 0.0, "first_click": 0.0, "linear": 0.0, "time_decay": 0.0}
        for canal in canaux_uniques
    }

    for client_id, parcours in df_conv.groupby("client_id"):
        parcours = parcours.sort_values("position")
        canaux   = parcours["canal"].tolist()
        n        = len(canaux)

        resultats[canaux[-1]]["last_click"]  += 1
        resultats[canaux[0]]["first_click"]  += 1

        for canal in canaux:
            resultats[canal]["linear"] += 1 / n

        poids = np.array([2 ** i for i in range(n)], dtype=float)
        poids = poids / poids.sum()
        for canal, p in zip(canaux, poids):
            resultats[canal]["time_decay"] += p

    df_attr = pd.DataFrame(resultats).T.reset_index()
    df_attr.columns = ["canal", "last_click", "first_click", "linear", "time_decay"]

    # Normalisation en pourcentages (somme = 100%)
    nb_conv = len(clients_convertis)
    if nb_conv > 0:
        df_attr["last_click"]  = (df_attr["last_click"]  / nb_conv * 100).round(2)
        df_attr["first_click"] = (df_attr["first_click"] / nb_conv * 100).round(2)
        df_attr["linear"]      = (df_attr["linear"]      / nb_conv * 100).round(2)
        df_attr["time_decay"]  = (df_attr["time_decay"]  / nb_conv * 100).round(2)

    df_attr = df_attr.sort_values("last_click", ascending=False)

    logger.info("Attribution calculée sur %d parcours convertis", nb_conv)
    return df_attr


# ── Stats canaux ──────────────────────────────────────────────────────────────

def compute_canal_stats(
    touchpoints: pd.DataFrame,
    clients: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calcule les statistiques de performance par canal.

    Métriques :
        - nb_last_touch   : nombre de fois que le canal est dernier touchpoint
        - nb_conversions  : nombre de conversions sur ce canal (last touch)
        - taux_conversion : nb_conversions / nb_last_touch * 100 (en %)
        - n_touches_moyen : longueur moyenne du parcours

    Args:
        touchpoints: DataFrame de touchpoints issu de clean_datasets()
        clients:     DataFrame clients (accepté pour compatibilité tests)

    Returns:
        pd.DataFrame: Statistiques par canal
    """
    last_touches = touchpoints[touchpoints["is_last_touch"] == True].copy()

    stats = (
        last_touches.groupby("canal")
        .agg(
            nb_last_touch=("client_id", "count"),
            nb_conversions=("converti", "sum"),
            n_touches_moyen=("n_touches_total", "mean"),
        )
        .reset_index()
    )

    stats["taux_conversion"] = (
        stats["nb_conversions"] / stats["nb_last_touch"] * 100
    ).round(2)
    stats["n_touches_moyen"] = stats["n_touches_moyen"].round(2)
    stats = stats.sort_values("taux_conversion", ascending=False)

    logger.info(
        "Stats canaux calculées — %d canaux | taux de conversion moyen : %.1f%%",
        len(stats),
        stats["nb_conversions"].sum() / stats["nb_last_touch"].sum() * 100,
    )
    return stats


# ── ROI ───────────────────────────────────────────────────────────────────────

def compute_roi(
    touchpoints: pd.DataFrame,
    clients: pd.DataFrame,
    canal_costs: pd.DataFrame = None,
    conversion_value: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calcule le CPA et le ROI par canal en croisant coûts, conversions
    et valeur moyenne par segment.

    Args:
        touchpoints:      DataFrame de touchpoints
        clients:          DataFrame clients (pour la distribution des segments)
        canal_costs:      DataFrame des coûts par canal (optionnel,
                          chargé depuis SQLite si non fourni)
        conversion_value: DataFrame des valeurs par segment (optionnel,
                          chargé depuis SQLite si non fourni)

    Returns:
        pd.DataFrame: Table roi_by_canal avec colonnes :
                      canal, nb_touchpoints, cout_total, nb_conversions,
                      valeur_generee, cpa, roi, type_facturation
    """
    from src.load import query_sqlite

    if canal_costs is None:
        canal_costs = query_sqlite("SELECT * FROM canal_costs")
    if conversion_value is None:
        conversion_value = query_sqlite("SELECT * FROM conversion_value")

    # Nombre de touchpoints par canal
    nb_touches_canal = (
        touchpoints.groupby("canal")
        .size()
        .reset_index(name="nb_touchpoints")
    )

    df = nb_touches_canal.merge(canal_costs, on="canal", how="left")
    df["cout_total"] = (
        df["nb_touchpoints"] * df["cout_par_touchpoint_moyen"]
    ).round(2)

    # Conversions par canal (last-touch)
    conversions = (
        touchpoints[touchpoints["is_last_touch"] == True]
        .groupby("canal")["converti"]
        .sum()
        .reset_index(name="nb_conversions")
    )
    df = df.merge(conversions, on="canal", how="left")

    # Valeur moyenne globale pondérée par la distribution des segments
    segment_dist = (
        clients.groupby("segment")
        .size()
        .reset_index(name="nb_clients")
    )
    segment_dist["poids"] = (
        segment_dist["nb_clients"] / segment_dist["nb_clients"].sum()
    )
    segment_dist = segment_dist.merge(conversion_value, on="segment", how="left")
    valeur_moyenne_globale = (
        segment_dist["valeur_conversion_moyenne"] * segment_dist["poids"]
    ).sum()

    df["valeur_generee"] = (df["nb_conversions"] * valeur_moyenne_globale).round(2)

    # CPA = coût total / nb conversions
    df["cpa"] = (
        df["cout_total"] / df["nb_conversions"].replace(0, np.nan)
    ).round(2)

    # ROI = (valeur générée - coût total) / coût total × 100
    df["roi"] = np.where(
        df["cout_total"] > 0,
        ((df["valeur_generee"] - df["cout_total"]) / df["cout_total"] * 100).round(1),
        np.nan,
    )

    df = df[[
        "canal", "nb_touchpoints", "cout_total", "nb_conversions",
        "valeur_generee", "cpa", "roi", "type_facturation",
    ]]
    df = df.sort_values("roi", ascending=False, na_position="first")

    meilleur = df.dropna(subset=["roi"]).iloc[0]["canal"] if df["roi"].notna().any() else "N/A"
    logger.info("ROI calculé par canal — canal le plus rentable : %s", meilleur)
    return df


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_transformations(datasets: dict) -> dict:
    """
    Orchestre l'ensemble des transformations du pipeline marketing.

    Args:
        datasets: dict {"clients": DataFrame, "touchpoints": DataFrame}

    Returns:
        dict: Données transformées avec clés :
              clients, touchpoints, attribution, canal_stats
    """
    logger.info("Démarrage des transformations...")

    data = clean_datasets(datasets)

    data["attribution"] = compute_attribution(data["touchpoints"])
    data["canal_stats"] = compute_canal_stats(data["touchpoints"], data["clients"])

    logger.info("Transformations terminées")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from src.extract import load_all_datasets
    datasets = load_all_datasets()
    result = run_transformations(datasets)
    logger.info("Attribution :\n%s", result["attribution"].to_string())