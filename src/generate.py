"""
generate.py
-----------
Simulation d'un dataset de parcours client multitouch réaliste.
Inspiré des patterns observés en marketing télécom/e-commerce.

Structure générée :
    - clients.csv      : 50 000 clients avec profil démographique
    - touchpoints.csv  : historique des interactions par client/canal

Méthode de génération de la conversion :
    La conversion est modélisée via un score logistique (sigmoid) combinant
    6 features métier pondérées. Cette approche garantit un signal fort et
    apprenable par le modèle ML, avec un taux de conversion cible ~20%.

    Score = sigmoid(w0 + w1*canal + w2*segment + w3*age + w4*anciennete
                    + w5*n_touches + w6*sequence)
"""

import numpy as np
import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
N_CLIENTS = 50_000

# ── Paramètres réalistes par canal (inspirés benchmarks marketing 2022-2024) ──
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
    "YouTube":    0,
    "Facebook":   1,
    "Instagram":  2,
    "SEO":        3,
    "Google Ads": 4,
    "Email":      5,
}

# Encodage des segments (0=faible → 3=fort)
SEGMENT_SCORE = {
    "Churner":   0,
    "Low-Value": 1,
    "Standard":  2,
    "Premium":   3,
}

DISCOVERY_CANAUX = {"SEO", "Google Ads", "YouTube"}
CLOSING_CANAUX   = {"Email"}


def _sigmoid(x: float) -> float:
    """Fonction sigmoid : transforme un score réel en probabilité [0, 1]."""
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

    Returns:
        float: Probabilité de conversion entre 0 et 1
    """
    # Normalisation de chaque feature sur une échelle comparable
    x_canal    = (CANAL_SCORE[canal_last] - 2.5) / 1.5        # [-1.67, +1.67]
    x_segment  = (SEGMENT_SCORE[segment] - 1.5) / 1.0         # [-1.50, +1.50]
    x_age      = -(abs(age - 35) - 10) / 10                   # peak à 35 ans
    x_anciente = 1.0 if anciennete_mois < 6 else (
                 0.5 if anciennete_mois > 60 else
                -0.5 if anciennete_mois > 24 else 0.0
    )
    x_touches  = (n_touches - 3) / 2                          # centré sur 3
    x_sequence = 1.0 if (
        len(canaux_parcours) >= 2
        and canaux_parcours[0] in DISCOVERY_CANAUX
        and canaux_parcours[-1] in CLOSING_CANAUX
    ) else 0.0

    # Score logistique — intercept calibré pour ~20% de taux de conversion
    score = (
        -1.80                              # intercept
        + 0.90 * x_canal                  # canal last-touch : signal fort
        + 0.75 * x_segment                # segment : signal fort
        + 0.40 * x_age                    # âge : signal modéré
        + 0.35 * x_anciente               # ancienneté : signal modéré
        + 0.30 * x_touches                # nb touchpoints : signal modéré
        + 0.25 * x_sequence               # séquence : signal faible
        + np.random.normal(0, 0.50)       # bruit résiduel réaliste
    )

    return _sigmoid(score)


def generate_clients() -> pd.DataFrame:
    """
    Génère la table clients avec profil démographique.

    Returns:
        pd.DataFrame: Table clients (50 000 lignes)
    """
    ages = np.random.normal(loc=38, scale=12, size=N_CLIENTS).clip(18, 75).astype(int)

    segments = np.random.choice(
        ["Premium", "Standard", "Low-Value", "Churner"],
        size=N_CLIENTS,
        p=[0.20, 0.45, 0.25, 0.10],
    )

    regions = np.random.choice(
        ["Île-de-France", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine",
         "Occitanie", "Hauts-de-France", "PACA", "Grand Est", "Autres"],
        size=N_CLIENTS,
        p=[0.28, 0.15, 0.10, 0.09, 0.09, 0.08, 0.07, 0.14],
    )

    clients = pd.DataFrame({
        "client_id":       range(1, N_CLIENTS + 1),
        "age":             ages,
        "segment":         segments,
        "region":          regions,
        "anciennete_mois": np.random.randint(1, 120, size=N_CLIENTS),
    })

    print(f"✅ Clients générés — {clients.shape[0]} lignes")
    return clients


def generate_touchpoints(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Génère la table des touchpoints (parcours client multitouch).

    Chaque client a entre 1 et 6 interactions sur une période de 90 jours.
    La probabilité de conversion est calculée via un score logistique (sigmoid)
    combinant 6 features métier — garantissant un signal fort et apprenable.

    Returns:
        pd.DataFrame: Table touchpoints
    """
    records = []
    date_debut = pd.Timestamp("2022-01-01")

    for _, client in clients.iterrows():
        client_id       = client["client_id"]
        segment         = client["segment"]
        age             = int(client["age"])
        anciennete_mois = int(client["anciennete_mois"])

        # Nombre de touchpoints (1 à 6)
        n_touches = np.random.choice(
            [1, 2, 3, 4, 5, 6],
            p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
        )

        # Sélection des canaux du parcours
        canaux_parcours = list(np.random.choice(CANAUX, size=n_touches, p=WEIGHTS))

        # Dates espacées aléatoirement sur 90 jours
        jours = sorted(np.random.choice(range(90), size=n_touches, replace=False))
        dates = [date_debut + pd.Timedelta(days=int(j)) for j in jours]

        # ── Probabilité de conversion via score logistique ──
        proba_conversion = _compute_conversion_proba(
            canal_last      = canaux_parcours[-1],
            segment         = segment,
            age             = age,
            anciennete_mois = anciennete_mois,
            n_touches       = n_touches,
            canaux_parcours = canaux_parcours,
        )
        converti = np.random.random() < proba_conversion

        # Enregistrement de chaque touchpoint
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
    print(f"✅ Touchpoints générés — {df.shape[0]} lignes")
    print(f"   Taux de conversion global : {df[df['is_last_touch']]['converti'].mean():.2%}")
    return df


def save_datasets(clients: pd.DataFrame, touchpoints: pd.DataFrame) -> None:
    """Sauvegarde les datasets générés en CSV."""
    clients.to_csv(RAW_DATA_PATH / "clients.csv", index=False)
    touchpoints.to_csv(RAW_DATA_PATH / "touchpoints.csv", index=False)
    print(f"✅ clients.csv sauvegardé ({len(clients)} lignes)")
    print(f"✅ touchpoints.csv sauvegardé ({len(touchpoints)} lignes)")


def run_generation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point d'entrée principal de la génération."""
    print("=== GÉNÉRATION DU DATASET ===")
    clients     = generate_clients()
    touchpoints = generate_touchpoints(clients)
    save_datasets(clients, touchpoints)
    return clients, touchpoints


if __name__ == "__main__":
    run_generation()