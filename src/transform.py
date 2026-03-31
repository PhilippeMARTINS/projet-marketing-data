"""
transform.py
------------
Module de transformation : nettoyage, enrichissement et calcul des modèles
d'attribution marketing (Last Click, First Click, Linear, Time Decay).
"""

import pandas as pd
import numpy as np


def clean_datasets(datasets: dict) -> dict:
    """
    Nettoie les datasets bruts.

    Args:
        datasets: dict {"clients": DataFrame, "touchpoints": DataFrame}

    Returns:
        dict: Datasets nettoyés
    """
    clients = datasets["clients"].copy()
    touchpoints = datasets["touchpoints"].copy()

    # Conversion des dates
    touchpoints["date"] = pd.to_datetime(touchpoints["date"])

    # Conversion des booléens
    touchpoints["is_first_touch"] = touchpoints["is_first_touch"].astype(bool)
    touchpoints["is_last_touch"] = touchpoints["is_last_touch"].astype(bool)

    print(f"✅ Nettoyage terminé")
    return {"clients": clients, "touchpoints": touchpoints}


def compute_attribution(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les modèles d'attribution marketing par canal.

    Modèles implémentés :
    - Last Click  : 100% du crédit au dernier canal
    - First Click : 100% du crédit au premier canal
    - Linear      : crédit équiréparti entre tous les canaux du parcours
    - Time Decay  : crédit pondéré par proximité à la conversion

    Returns:
        pd.DataFrame: Attribution par canal pour chaque modèle
    """

    # ── Récupération des parcours convertis uniquement ──────────────────────
    # On identifie les clients qui ont converti
    clients_convertis = (
        touchpoints[touchpoints["converti"] == 1]["client_id"].unique()
    )

    df_conv = touchpoints[touchpoints["client_id"].isin(clients_convertis)].copy()

    resultats = {canal: {"last_click": 0, "first_click": 0,
                          "linear": 0, "time_decay": 0}
                 for canal in df_conv["canal"].unique()}

    # ── Calcul par client converti ───────────────────────────────────────────
    for client_id, parcours in df_conv.groupby("client_id"):
        parcours = parcours.sort_values("position")
        canaux = parcours["canal"].tolist()
        n = len(canaux)

        # Last Click
        resultats[canaux[-1]]["last_click"] += 1

        # First Click
        resultats[canaux[0]]["first_click"] += 1

        # Linear : crédit équiréparti
        for canal in canaux:
            resultats[canal]["linear"] += 1 / n

        # Time Decay : poids exponentiel croissant vers la fin
        poids = np.array([2 ** i for i in range(n)], dtype=float)
        poids = poids / poids.sum()
        for canal, p in zip(canaux, poids):
            resultats[canal]["time_decay"] += p

    df_attr = pd.DataFrame(resultats).T.reset_index()
    df_attr.columns = ["canal", "last_click", "first_click", "linear", "time_decay"]

    # Normalisation en pourcentages
    for col in ["last_click", "first_click", "linear", "time_decay"]:
        df_attr[col] = (df_attr[col] / df_attr[col].sum() * 100).round(2)

    df_attr = df_attr.sort_values("last_click", ascending=False)

    print(f"✅ Attribution calculée pour {len(clients_convertis)} clients convertis")
    print(f"\n📊 Attribution par canal (%) :\n{df_attr.to_string(index=False)}")
    return df_attr


def compute_canal_stats(touchpoints: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques de performance par canal :
    - Taux de conversion last-touch
    - Nombre moyen de touchpoints avant conversion
    - Position moyenne dans le parcours

    Returns:
        pd.DataFrame: Stats par canal
    """
    # Jointure touchpoints + clients pour avoir le segment
    df = touchpoints.merge(clients[["client_id", "segment"]], on="client_id", how="left")

    # Stats sur les last touchpoints uniquement (= point de conversion)
    last_touches = df[df["is_last_touch"]].copy()

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

    print(f"\n📊 Stats par canal :\n{stats.to_string(index=False)}")
    return stats


def run_transformations(datasets: dict) -> dict:
    """
    Orchestre toutes les transformations.

    Returns:
        dict: {
            "clients": DataFrame,
            "touchpoints": DataFrame,
            "attribution": DataFrame,
            "canal_stats": DataFrame
        }
    """
    datasets = clean_datasets(datasets)
    touchpoints = datasets["touchpoints"]
    clients = datasets["clients"]

    attribution = compute_attribution(touchpoints)
    canal_stats = compute_canal_stats(touchpoints, clients)

    return {
        "clients": clients,
        "touchpoints": touchpoints,
        "attribution": attribution,
        "canal_stats": canal_stats,
    }