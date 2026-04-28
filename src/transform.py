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

def compute_roi(touchpoints: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le CPA et le ROI par canal en croisant :
    - Les coûts par touchpoint (canal_costs)
    - Les conversions par canal (canal_stats)
    - La valeur moyenne par conversion et par segment (conversion_value)

    Returns:
        pd.DataFrame: Table roi_by_canal avec CPA, ROI, coût total, valeur générée
    """
    from src.load import query_sqlite

    # Chargement des tables de référence
    canal_costs      = query_sqlite("SELECT * FROM canal_costs")
    conversion_value = query_sqlite("SELECT * FROM conversion_value")

    # Nombre de touchpoints par canal
    nb_touches_canal = (
        touchpoints.groupby("canal")
        .size()
        .reset_index(name="nb_touchpoints")
    )

    # Coût total par canal = nb touchpoints × coût moyen par touchpoint
    df = nb_touches_canal.merge(canal_costs, on="canal", how="left")
    df["cout_total"] = (
        df["nb_touchpoints"] * df["cout_par_touchpoint_moyen"]
    ).round(2)

    # Nombre de conversions par canal (last-touch)
    conversions = (
        touchpoints[touchpoints["is_last_touch"] == True]
        .groupby("canal")["converti"]
        .sum()
        .reset_index(name="nb_conversions")
    )
    df = df.merge(conversions, on="canal", how="left")

    # Valeur générée = nb conversions × valeur moyenne pondérée par segment
    # On calcule la valeur moyenne globale pondérée par la distribution des segments
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

    df["valeur_generee"] = (
        df["nb_conversions"] * valeur_moyenne_globale
    ).round(2)

    # CPA = coût total / nb conversions
    df["cpa"] = (
        df["cout_total"] / df["nb_conversions"].replace(0, np.nan)
    ).round(2)

    # ROI = (valeur générée - coût total) / coût total × 100
    # SEO a un coût nul → ROI infini → on le gère séparément
    df["roi"] = np.where(
        df["cout_total"] > 0,
        ((df["valeur_generee"] - df["cout_total"]) / df["cout_total"] * 100).round(1),
        np.nan,  # SEO : organique, ROI non calculable de cette façon
    )

    df = df[["canal", "nb_touchpoints", "cout_total", "nb_conversions",
             "valeur_generee", "cpa", "roi", "type_facturation"]]
    df = df.sort_values("roi", ascending=False, na_position="first")

    print(f"\n📊 ROI par canal :")
    print(df.to_string(index=False))
    return df

def run_transformations(datasets: dict) -> dict:
    """Orchestre toutes les transformations."""
    datasets   = clean_datasets(datasets)
    touchpoints = datasets["touchpoints"]
    clients     = datasets["clients"]

    attribution = compute_attribution(touchpoints)
    canal_stats = compute_canal_stats(touchpoints, clients)
    roi         = compute_roi(touchpoints, clients)  # ← nouveau

    return {
        "clients":    clients,
        "touchpoints": touchpoints,
        "attribution": attribution,
        "canal_stats": canal_stats,
        "roi":         roi,          # ← nouveau
    }