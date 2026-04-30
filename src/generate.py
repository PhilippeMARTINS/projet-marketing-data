"""
generate.py
-----------
Module de génération : simulation du dataset marketing.
Produit 50 000 clients et ~150 000 touchpoints avec des patterns réalistes
inspirés des benchmarks marketing digitaux français 2023-2024.

Méthode de génération de la conversion :
    La conversion est modélisée via un score logistique (sigmoid) combinant
    6 features métier pondérées. Cette approche garantit un signal fort et
    apprenable par le modèle ML.
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

N_CLIENTS = 50_000

# Paramètres par canal
CANAL_PARAMS = {
    "Email":      {"weight": 0.25, "position": "last"},
    "SEO":        {"weight": 0.20, "position": "first"},
    "Google Ads": {"weight": 0.20, "position": "first"},
    "Instagram":  {"weight": 0.15, "position": "middle"},
    "Facebook":   {"weight": 0.12, "position": "middle"},
    "YouTube":    {"weight": 0.08, "position": "first"},
}

CANAUX  = list(CANAL_PARAMS.keys())
WEIGHTS = [CANAL_PARAMS[c]["weight"] for c in CANAUX]

# Encodage ordonné des canaux par performance last-touch (0=faible → 5=fort)
CANAL_SCORE = {
    "YouTube": 0, "Facebook": 1, "Instagram": 2,
    "SEO": 3, "Google Ads": 4, "Email": 5,
}

# Encodage des segments (0=faible → 3=fort)
SEGMENT_SCORE = {
    "Churner": 0, "Low-Value": 1, "Standard": 2, "Premium": 3,
}

DISCOVERY_CANAUX = {"SEO", "Google Ads", "YouTube"}
CLOSING_CANAUX   = {"Email"}

REGIONS = [
    "Île-de-France", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine",
    "Occitanie", "Hauts-de-France", "PACA", "Grand Est", "Autres",
]


# ── Fonctions utilitaires ─────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Fonction sigmoïde : transforme un score réel en probabilité entre 0 et 1.

    Args:
        x: Score réel quelconque

    Returns:
        float: Probabilité dans ]0, 1[
    """
    return 1 / (1 + np.exp(-x))


def _compute_conversion_proba(
    canal_last: str,
    segment: str,
    age: int,
    anciennete_mois: int,
    n_touches: int,
    canaux_parcours: list,
) -> float:
    """
    Calcule la probabilité de conversion via un score logistique.

    Poids calibrés pour obtenir un taux de conversion global ~20%
    avec une AUC modèle cible de 0.75-0.80.

    Args:
        canal_last:      Dernier canal du parcours
        segment:         Segment client (Premium, Standard, Low-Value, Churner)
        age:             Âge du client
        anciennete_mois: Ancienneté en mois
        n_touches:       Nombre total de touchpoints
        canaux_parcours: Liste ordonnée des canaux du parcours

    Returns:
        float: Probabilité de conversion entre 0 et 1
    """
    # Normalisation de chaque feature sur une échelle comparable
    x_canal    = (CANAL_SCORE[canal_last] - 2.5) / 1.5
    x_segment  = (SEGMENT_SCORE[segment] - 1.5) / 1.0
    x_age      = -(abs(age - 35) - 10) / 10
    x_anciente = (
        1.0  if anciennete_mois < 6  else
        0.5  if anciennete_mois > 60 else
        -0.5 if anciennete_mois > 24 else
        0.0
    )
    x_touches  = (n_touches - 3) / 2
    x_sequence = 1.0 if (
        len(canaux_parcours) >= 2
        and canaux_parcours[0] in DISCOVERY_CANAUX
        and canaux_parcours[-1] in CLOSING_CANAUX
    ) else 0.0

    # Score logistique — intercept calibré pour ~20% de taux de conversion
    score = (
        -1.80
        + 0.90 * x_canal
        + 0.75 * x_segment
        + 0.40 * x_age
        + 0.35 * x_anciente
        + 0.30 * x_touches
        + 0.25 * x_sequence
        + np.random.normal(0, 0.50)
    )
    return _sigmoid(score)


# ── Générateurs ───────────────────────────────────────────────────────────────

def generate_clients() -> pd.DataFrame:
    """
    Génère la table clients avec segments, régions et attributs démographiques.

    Returns:
        pd.DataFrame: 50 000 clients simulés
    """
    ages = np.random.normal(loc=38, scale=12, size=N_CLIENTS).clip(18, 75).astype(int)

    segments = np.random.choice(
        ["Premium", "Standard", "Low-Value", "Churner"],
        size=N_CLIENTS,
        p=[0.20, 0.45, 0.25, 0.10],
    )

    clients = pd.DataFrame({
        "client_id":       range(1, N_CLIENTS + 1),
        "age":             ages,
        "segment":         segments,
        "region":          np.random.choice(REGIONS, N_CLIENTS,
                                            p=[0.28, 0.15, 0.10, 0.09,
                                               0.09, 0.08, 0.07, 0.14]),
        "anciennete_mois": np.random.randint(1, 120, size=N_CLIENTS),
    })

    logger.info(
        "Clients générés — %d lignes | segments : %s",
        len(clients),
        clients["segment"].value_counts().to_dict(),
    )
    return clients


def generate_touchpoints(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Génère la table touchpoints : parcours multi-canal pour chaque client.

    Chaque client a entre 1 et 6 touchpoints. La probabilité de conversion
    est calculée via _compute_conversion_proba() (score logistique).

    Args:
        clients: DataFrame issu de generate_clients()

    Returns:
        pd.DataFrame: Touchpoints avec colonnes de conversion
    """
    records    = []
    date_debut = pd.Timestamp("2022-01-01")

    for _, client in clients.iterrows():
        client_id       = client["client_id"]
        segment         = client["segment"]
        age             = int(client["age"])
        anciennete_mois = int(client["anciennete_mois"])

        n_touches = np.random.choice(
            [1, 2, 3, 4, 5, 6],
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05],
        )
        canaux_parcours = list(np.random.choice(CANAUX, size=n_touches, p=WEIGHTS))
        jours = sorted(np.random.choice(range(90), size=n_touches, replace=False))
        dates = [date_debut + pd.Timedelta(days=int(j)) for j in jours]

        proba_conversion = _compute_conversion_proba(
            canal_last=canaux_parcours[-1],
            segment=segment,
            age=age,
            anciennete_mois=anciennete_mois,
            n_touches=n_touches,
            canaux_parcours=canaux_parcours,
        )
        converti = np.random.random() < proba_conversion

        for i, (canal, date) in enumerate(zip(canaux_parcours, dates)):
            position = i + 1
            is_last  = position == n_touches
            records.append({
                "client_id":       client_id,
                "touchpoint_id":   len(records) + 1,
                "canal":           canal,
                "date":            date.date(),
                "position":        position,
                "n_touches_total": n_touches,
                "is_first_touch":  position == 1,
                "is_last_touch":   is_last,
                "converti":        int(converti) if is_last else 0,
            })

    df = pd.DataFrame(records)
    taux = df[df["is_last_touch"]]["converti"].mean()
    logger.info(
        "Touchpoints générés — %d lignes | taux de conversion : %.1f%%",
        len(df), taux * 100,
    )
    return df


def generate_canal_costs() -> pd.DataFrame:
    """
    Génère la table des coûts moyens par touchpoint et par canal.

    Returns:
        pd.DataFrame: Coûts par canal
    """
    data = {
        "canal": ["Email", "SEO", "Google Ads", "Instagram", "Facebook", "YouTube"],
        "cout_par_touchpoint_moyen": [0.03, 0.00, 1.65, 1.30, 1.15, 0.30],
        "cout_par_touchpoint_min":   [0.01, 0.00, 0.80, 0.60, 0.50, 0.10],
        "cout_par_touchpoint_max":   [0.05, 0.00, 2.50, 2.00, 1.80, 0.50],
        "type_facturation": ["Email", "Organique", "CPC", "CPM", "CPM", "CPV"],
    }
    df = pd.DataFrame(data)
    logger.info("Canal costs générés — %d canaux", len(df))
    return df


def generate_conversion_value() -> pd.DataFrame:
    """
    Génère la table de valeur estimée par conversion et par segment.

    Returns:
        pd.DataFrame: Valeur de conversion par segment
    """
    data = {
        "segment": ["Premium", "Standard", "Low-Value", "Churner"],
        "valeur_conversion_moyenne": [180.0, 85.0, 35.0, 20.0],
        "valeur_conversion_min":     [120.0, 50.0, 15.0, 10.0],
        "valeur_conversion_max":     [350.0, 150.0, 60.0, 40.0],
    }
    df = pd.DataFrame(data)
    logger.info("Conversion value générée — %d segments", len(df))
    return df


def save_datasets(
    clients: pd.DataFrame,
    touchpoints: pd.DataFrame,
    canal_costs: pd.DataFrame,
    conversion_value: pd.DataFrame,
) -> None:
    """
    Sauvegarde les 4 datasets générés en CSV dans data/raw/.

    Args:
        clients:          Table clients
        touchpoints:      Table touchpoints
        canal_costs:      Table des coûts par canal
        conversion_value: Table des valeurs de conversion par segment
    """
    clients.to_csv(RAW_DATA_PATH / "clients.csv", index=False)
    touchpoints.to_csv(RAW_DATA_PATH / "touchpoints.csv", index=False)
    canal_costs.to_csv(RAW_DATA_PATH / "canal_costs.csv", index=False)
    conversion_value.to_csv(RAW_DATA_PATH / "conversion_value.csv", index=False)

    logger.info(
        "Datasets sauvegardés dans '%s' : clients (%d), touchpoints (%d), "
        "canal_costs (%d), conversion_value (%d)",
        RAW_DATA_PATH, len(clients), len(touchpoints),
        len(canal_costs), len(conversion_value),
    )


def run_generation() -> tuple:
    """
    Orchestre la génération complète du dataset marketing.

    Returns:
        tuple: (clients, touchpoints, canal_costs, conversion_value)
    """
    logger.info("Génération du dataset marketing simulé...")
    clients          = generate_clients()
    touchpoints      = generate_touchpoints(clients)
    canal_costs      = generate_canal_costs()
    conversion_value = generate_conversion_value()
    save_datasets(clients, touchpoints, canal_costs, conversion_value)
    logger.info("Génération terminée")
    return clients, touchpoints, canal_costs, conversion_value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_generation()