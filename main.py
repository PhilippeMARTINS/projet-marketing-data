"""
main.py
-------
Point d'entrée du pipeline complet — Marketing Parcours Client Multitouch.
Generate -> Extract -> Validate -> Transform -> ROI -> Validate -> Load -> Analyze -> Model
"""

import logging
from src.generate import run_generation
from src.extract import load_all_datasets
from src.transform import run_transformations, compute_roi
from src.load import save_to_sqlite
from src.analyze import run_analysis
from src.model import run_model
from src.validate import validate_raw_data, validate_transformed_data


# ── Configuration du logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  PIPELINE MARKETING — PARCOURS CLIENT MULTITOUCH")
    logger.info("=" * 50)

    logger.info("ETAPE 1 — GENERATION DES DONNEES")
    clients, touchpoints, canal_costs, conversion_value = run_generation()

    logger.info("ETAPE 2 — EXTRACTION")
    datasets = load_all_datasets()

    logger.info("ETAPE 3 — VALIDATION DONNEES BRUTES")
    validate_raw_data(datasets)

    logger.info("ETAPE 4 — TRANSFORMATION")
    data = run_transformations(datasets)

    logger.info("ETAPE 5 — CALCUL ROI")
    roi = compute_roi(
        touchpoints=data["touchpoints"],
        clients=data["clients"],
        canal_costs=canal_costs,
        conversion_value=conversion_value,
    )
    data["roi"]              = roi
    data["canal_costs"]      = canal_costs
    data["conversion_value"] = conversion_value

    logger.info("ETAPE 6 — VALIDATION DONNEES TRANSFORMEES")
    validate_transformed_data(data)

    logger.info("ETAPE 7 — CHARGEMENT SQL")
    save_to_sqlite(data)

    logger.info("ETAPE 8 — ANALYSE & VISUALISATION")
    run_analysis()

    logger.info("ETAPE 9 — MODELE ML")
    run_model()

    logger.info("=" * 50)
    logger.info("PIPELINE TERMINE")
    logger.info("=" * 50)