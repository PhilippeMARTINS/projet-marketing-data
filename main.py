"""
main.py
-------
Point d'entrée du pipeline complet :
Generate → Extract → Transform → Load → Analyze → Model
"""

import logging
from src.generate import run_generation
from src.extract import load_all_datasets
from src.transform import run_transformations
from src.load import save_to_sqlite
from src.analyze import run_analysis
from src.model import run_model


# ── Configuration du logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                        # affichage console
        logging.FileHandler("pipeline.log", mode="w"), # sauvegarde dans un fichier
    ],
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  PIPELINE MARKETING — PARCOURS CLIENT MULTITOUCH")
    logger.info("=" * 50)

    logger.info("ÉTAPE 1 — GÉNÉRATION DES DONNÉES")
    run_generation()

    logger.info("ÉTAPE 2 — EXTRACTION")
    datasets = load_all_datasets()

    logger.info("ÉTAPE 3 — TRANSFORMATION")
    data = run_transformations(datasets)

    logger.info("ÉTAPE 4 — CHARGEMENT SQL")
    save_to_sqlite(data)

    logger.info("ÉTAPE 5 — ANALYSE & VISUALISATION")
    run_analysis()

    logger.info("ÉTAPE 6 — MODÈLE ML")
    run_model()

    logger.info("=" * 50)
    logger.info("PIPELINE TERMINÉ")
    logger.info("=" * 50)