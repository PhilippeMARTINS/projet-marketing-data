"""
main.py
-------
Point d'entrée du pipeline complet :
Generate → Extract → Transform → Load → Analyze → Model
"""

from src.generate import run_generation
from src.extract import load_all_datasets
from src.transform import run_transformations
from src.load import save_to_sqlite
from src.analyze import run_analysis
from src.model import run_model


if __name__ == "__main__":
    print("=" * 50)
    print("  PIPELINE MARKETING — PARCOURS CLIENT MULTITOUCH")
    print("=" * 50)

    print("\n=== ÉTAPE 1 — GÉNÉRATION DES DONNÉES ===")
    run_generation()

    print("\n=== ÉTAPE 2 — EXTRACTION ===")
    datasets = load_all_datasets()

    print("\n=== ÉTAPE 3 — TRANSFORMATION ===")
    data = run_transformations(datasets)

    print("\n=== ÉTAPE 4 — CHARGEMENT SQL ===")
    save_to_sqlite(data)

    print("\n=== ÉTAPE 5 — ANALYSE & VISUALISATION ===")
    run_analysis()

    print("\n=== ÉTAPE 6 — MODÈLE ML ===")
    run_model()

    print("\n" + "=" * 50)
    print("  PIPELINE TERMINÉ")
    print("=" * 50)