"""
extract.py
----------
Module d'extraction : chargement des datasets clients et touchpoints
depuis les fichiers CSV générés par generate.py.
"""

import logging
from pathlib import Path

import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw")


def load_clients() -> pd.DataFrame:
    """
    Charge la table clients depuis data/raw/clients.csv.

    Returns:
        pd.DataFrame: Table clients

    Raises:
        FileNotFoundError: Si clients.csv est absent (lancer d'abord generate.py)
    """
    chemin = RAW_DATA_PATH / "clients.csv"
    if not chemin.exists():
        logger.error("Fichier manquant : %s — lancer d'abord generate.py", chemin)
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)
    logger.info("clients chargé — %d lignes, %d colonnes", df.shape[0], df.shape[1])
    return df


def load_touchpoints() -> pd.DataFrame:
    """
    Charge la table touchpoints depuis data/raw/touchpoints.csv.

    Returns:
        pd.DataFrame: Table touchpoints

    Raises:
        FileNotFoundError: Si touchpoints.csv est absent (lancer d'abord generate.py)
    """
    chemin = RAW_DATA_PATH / "touchpoints.csv"
    if not chemin.exists():
        logger.error("Fichier manquant : %s — lancer d'abord generate.py", chemin)
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)
    logger.info("touchpoints chargé — %d lignes, %d colonnes", df.shape[0], df.shape[1])
    return df


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Charge tous les datasets du projet marketing.

    Returns:
        dict: {"clients": DataFrame, "touchpoints": DataFrame}

    Raises:
        FileNotFoundError: Si un fichier CSV est absent
    """
    logger.info("Chargement des datasets marketing depuis '%s'", RAW_DATA_PATH)
    datasets = {
        "clients":     load_clients(),
        "touchpoints": load_touchpoints(),
    }
    logger.info("Extraction terminée — %d datasets chargés", len(datasets))
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    datasets = load_all_datasets()
    logger.info("Aperçu touchpoints :\n%s", datasets["touchpoints"].head().to_string())