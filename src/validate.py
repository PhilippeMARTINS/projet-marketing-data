"""
validate.py
-----------
Validation de la qualité des données générées et transformées.
Appelé automatiquement par main.py après chaque étape clé du pipeline.
"""

import logging
import pandas as pd


logger = logging.getLogger(__name__)

# ── Constantes de référence ───────────────────────────────────────────────────
SEGMENTS_ATTENDUS = {"Premium", "Standard", "Low-Value", "Churner"}
CANAUX_ATTENDUS   = {"Email", "Google Ads", "SEO", "Instagram", "Facebook", "YouTube"}
PART_PREMIUM      = 0.20
PART_CHURNER      = 0.10
TOLERANCE         = 0.05
TAUX_CONVERSION_MIN = 0.05
TAUX_CONVERSION_MAX = 0.60


def _check(label: str, condition: bool, detail: str = "") -> bool:
    """
    Logue le résultat d'un check et retourne le succès.

    Args:
        label:     Libellé du check
        condition: True si le check passe
        detail:    Message complémentaire en cas d'échec

    Returns:
        bool: True si le check passe, False sinon
    """
    if condition:
        logger.info("    [OK] %s", label)
    else:
        msg = f"    [KO] {label}"
        if detail:
            msg += f" -- {detail}"
        logger.warning(msg)
    return condition


def validate_raw_data(datasets: dict) -> bool:
    """
    Valide les données brutes après génération et extraction.

    Args:
        datasets: Dictionnaire {"clients": DataFrame, "touchpoints": DataFrame}

    Returns:
        bool: True si toutes les validations passent
    """
    logger.info("Validation des donnees brutes :")
    all_passed = True

    # ── clients ───────────────────────────────────────────────────────────────
    logger.info("  [clients]")
    clients = datasets["clients"]

    all_passed &= _check(
        "client_id sans doublons",
        clients["client_id"].nunique() == len(clients),
        f"{len(clients) - clients['client_id'].nunique()} doublons detectes",
    )
    all_passed &= _check(
        "segments dans les valeurs attendues",
        set(clients["segment"].unique()).issubset(SEGMENTS_ATTENDUS),
        f"valeurs inattendues : {set(clients['segment'].unique()) - SEGMENTS_ATTENDUS}",
    )

    proportions  = clients["segment"].value_counts(normalize=True)
    part_premium = proportions.get("Premium", 0)
    part_churner = proportions.get("Churner", 0)

    all_passed &= _check(
        f"proportion Premium entre {PART_PREMIUM - TOLERANCE:.0%} et {PART_PREMIUM + TOLERANCE:.0%}",
        abs(part_premium - PART_PREMIUM) <= TOLERANCE,
        f"observe : {part_premium:.1%}",
    )
    all_passed &= _check(
        f"proportion Churner entre {PART_CHURNER - TOLERANCE:.0%} et {PART_CHURNER + TOLERANCE:.0%}",
        abs(part_churner - PART_CHURNER) <= TOLERANCE,
        f"observe : {part_churner:.1%}",
    )
    all_passed &= _check(
        "age entre 18 et 75 ans",
        clients["age"].between(18, 75).all(),
        f"min={clients['age'].min()}, max={clients['age'].max()}",
    )
    all_passed &= _check(
        "anciennete_mois positif",
        (clients["anciennete_mois"] >= 0).all(),
    )

    # ── touchpoints ───────────────────────────────────────────────────────────
    logger.info("  [touchpoints]")
    tp = datasets["touchpoints"]

    all_passed &= _check(
        "canaux dans les valeurs attendues",
        set(tp["canal"].unique()).issubset(CANAUX_ATTENDUS),
        f"valeurs inattendues : {set(tp['canal'].unique()) - CANAUX_ATTENDUS}",
    )
    taux = tp["converti"].mean()
    all_passed &= _check(
        f"taux de conversion entre {TAUX_CONVERSION_MIN:.0%} et {TAUX_CONVERSION_MAX:.0%}",
        TAUX_CONVERSION_MIN <= taux <= TAUX_CONVERSION_MAX,
        f"observe : {taux:.1%}",
    )
    all_passed &= _check(
        "position >= 1",
        (tp["position"] >= 1).all(),
        f"min={tp['position'].min()}",
    )
    all_passed &= _check(
        "n_touches_total >= 1",
        (tp["n_touches_total"] >= 1).all(),
    )
    all_passed &= _check(
        "tous les clients du touchpoints existent dans clients",
        tp["client_id"].isin(clients["client_id"]).all(),
        f"{(~tp['client_id'].isin(clients['client_id'])).sum()} client_id orphelins",
    )

    if all_passed:
        logger.info("Validation donnees brutes : toutes les verifications sont passees [OK]")
    else:
        logger.warning("Validation donnees brutes : certains checks ont echoue [KO]")

    return all_passed


def validate_transformed_data(data: dict) -> bool:
    """
    Valide les données après transformation.

    Args:
        data: Dictionnaire issu de run_transformations()
              Cles attendues : touchpoints, canal_stats, attribution

    Returns:
        bool: True si toutes les validations passent
    """
    logger.info("Validation des donnees transformees :")
    all_passed = True

    # ── attribution — colonnes : last_click, first_click, linear, time_decay ──
    if "attribution" in data:
        logger.info("  [attribution]")
        attr    = data["attribution"]
        modeles = ["last_click", "first_click", "linear", "time_decay"]
        for modele in modeles:
            all_passed &= _check(f"colonne '{modele}' presente", modele in attr.columns)
        if all(m in attr.columns for m in modeles):
            somme = attr[modeles].sum().sum()
            all_passed &= _check(
                "somme des credits d'attribution > 0",
                somme > 0,
                f"somme totale : {somme:.2f}",
            )

    # ── canal_stats — taux_conversion en % (0-100) ────────────────────────────
    if "canal_stats" in data:
        logger.info("  [canal_stats]")
        stats = data["canal_stats"]
        all_passed &= _check(
            "taux_conversion entre 0% et 100%",
            stats["taux_conversion"].between(0, 100).all(),
            f"min={stats['taux_conversion'].min():.1f}, max={stats['taux_conversion'].max():.1f}",
        )

    if all_passed:
        logger.info("Validation donnees transformees : toutes les verifications sont passees [OK]")
    else:
        logger.warning("Validation donnees transformees : certains checks ont echoue [KO]")

    return all_passed