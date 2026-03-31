"""
extract.py
----------
Module d'extraction : chargement des datasets clients et touchpoints.
"""

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw")


def load_clients() -> pd.DataFrame:
    """Charge la table clients."""
    df = pd.read_csv(RAW_DATA_PATH / "clients.csv")
    print(f"✅ clients chargé — {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def load_touchpoints() -> pd.DataFrame:
    """Charge la table touchpoints."""
    df = pd.read_csv(RAW_DATA_PATH / "touchpoints.csv")
    print(f"✅ touchpoints chargé — {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Charge tous les datasets du projet.

    Returns:
        dict: {"clients": DataFrame, "touchpoints": DataFrame}
    """
    print("=== EXTRACTION ===")
    datasets = {
        "clients": load_clients(),
        "touchpoints": load_touchpoints(),
    }
    return datasets


if __name__ == "__main__":
    datasets = load_all_datasets()
    print("\n📋 Aperçu touchpoints :")
    print(datasets["touchpoints"].head())