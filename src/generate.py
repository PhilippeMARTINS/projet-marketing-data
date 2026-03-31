"""
generate.py
-----------
Simulation d'un dataset de parcours client multitouch réaliste.
Inspiré des patterns observés en marketing télécom/e-commerce.

Structure générée :
    - clients.csv      : 50 000 clients avec profil démographique
    - touchpoints.csv  : historique des interactions par client/canal
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
    "Email": {
        "weight": 0.25,          # fréquence d'apparition dans les parcours
        "conversion_boost": 0.18, # probabilité de conversion si last touch
        "position": "last",       # canal plutôt en fin de parcours
    },
    "SEO": {
        "weight": 0.20,
        "conversion_boost": 0.12,
        "position": "first",      # canal plutôt en début de parcours
    },
    "Google Ads": {
        "weight": 0.20,
        "conversion_boost": 0.14,
        "position": "first",
    },
    "Instagram": {
        "weight": 0.15,
        "conversion_boost": 0.08,
        "position": "middle",
    },
    "Facebook": {
        "weight": 0.12,
        "conversion_boost": 0.07,
        "position": "middle",
    },
    "YouTube": {
        "weight": 0.08,
        "conversion_boost": 0.05,
        "position": "first",
    },
}

CANAUX = list(CANAL_PARAMS.keys())
WEIGHTS = [CANAL_PARAMS[c]["weight"] for c in CANAUX]
CONVERSION_BOOST = {c: CANAL_PARAMS[c]["conversion_boost"] for c in CANAUX}


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
        "client_id": range(1, N_CLIENTS + 1),
        "age": ages,
        "segment": segments,
        "region": regions,
        "anciennete_mois": np.random.randint(1, 120, size=N_CLIENTS),
    })

    print(f"✅ Clients générés — {clients.shape[0]} lignes")
    return clients


def generate_touchpoints(clients: pd.DataFrame) -> pd.DataFrame:
    """
    Génère la table des touchpoints (parcours client multitouch).
    Chaque client a entre 1 et 6 interactions sur une période de 90 jours.
    La conversion dépend du dernier canal touché et du segment client.

    Returns:
        pd.DataFrame: Table touchpoints
    """
    records = []

    # Bonus de conversion par segment client
    segment_bonus = {
        "Premium": 0.15,
        "Standard": 0.05,
        "Low-Value": -0.05,
        "Churner": -0.10,
    }

    date_debut = pd.Timestamp("2022-01-01")

    for _, client in clients.iterrows():
        client_id = client["client_id"]
        segment = client["segment"]

        # Nombre de touchpoints pour ce client (1 à 6)
        n_touches = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])

        # Sélection des canaux du parcours
        canaux_parcours = np.random.choice(CANAUX, size=n_touches, p=WEIGHTS)

        # Dates des touchpoints (espacées aléatoirement sur 90 jours)
        jours = sorted(np.random.choice(range(90), size=n_touches, replace=False))
        dates = [date_debut + pd.Timedelta(days=int(j)) for j in jours]

        # Calcul de la conversion basée sur le dernier canal
        dernier_canal = canaux_parcours[-1]
        proba_conversion = (
            CONVERSION_BOOST[dernier_canal]
            + segment_bonus[segment]
        )
        proba_conversion = np.clip(proba_conversion, 0.01, 0.95)
        converti = np.random.random() < proba_conversion

        # Enregistrement de chaque touchpoint
        for i, (canal, date) in enumerate(zip(canaux_parcours, dates)):
            position = i + 1
            is_last = position == n_touches

            records.append({
                "client_id": client_id,
                "touchpoint_id": len(records) + 1,
                "canal": canal,
                "date": date.date(),
                "position": position,
                "n_touches_total": n_touches,
                "is_first_touch": position == 1,
                "is_last_touch": is_last,
                "converti": int(converti) if is_last else 0,
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
    clients = generate_clients()
    touchpoints = generate_touchpoints(clients)
    save_datasets(clients, touchpoints)
    return clients, touchpoints


if __name__ == "__main__":
    run_generation()