"""
test_transform.py
-----------------
Tests unitaires pour src/transform.py.

Fonctions testées :
    - clean_datasets        : nettoyage des types de données
    - compute_attribution   : calcul des 4 modèles d'attribution
    - compute_canal_stats   : statistiques de performance par canal

Lancer avec : pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from src.transform import clean_datasets, compute_attribution, compute_canal_stats


# ── Fixtures : données de test réutilisables ──────────────────────────────────

@pytest.fixture
def sample_clients() -> pd.DataFrame:
    """Table clients minimale pour les tests."""
    return pd.DataFrame({
        "client_id": [1, 2, 3],
        "age":       [30, 45, 25],
        "segment":   ["Premium", "Standard", "Churner"],
        "region":    ["Île-de-France", "PACA", "Occitanie"],
        "anciennete_mois": [12, 60, 3],
    })


@pytest.fixture
def sample_touchpoints() -> pd.DataFrame:
    """
    Table touchpoints minimale pour les tests.

    Parcours :
        Client 1 (converti)  : SEO → Email (2 touches)
        Client 2 (converti)  : Google Ads → Instagram → Email (3 touches)
        Client 3 (non converti) : Facebook (1 touche)
    """
    return pd.DataFrame({
        "client_id":       [1, 1, 2, 2, 2, 3],
        "touchpoint_id":   [1, 2, 3, 4, 5, 6],
        "canal":           ["SEO", "Email", "Google Ads", "Instagram", "Email", "Facebook"],
        "date":            ["2022-01-01", "2022-01-05",
                            "2022-01-02", "2022-01-04", "2022-01-08",
                            "2022-01-03"],
        "position":        [1, 2, 1, 2, 3, 1],
        "n_touches_total": [2, 2, 3, 3, 3, 1],
        "is_first_touch":  [True, False, True, False, False, True],
        "is_last_touch":   [False, True, False, False, True, True],
        "converti":        [0, 1, 0, 0, 1, 0],
    })


# ── Tests : clean_datasets ────────────────────────────────────────────────────

class TestCleanDatasets:
    """Tests pour la fonction clean_datasets."""

    def test_date_converted_to_datetime(self, sample_clients, sample_touchpoints):
        """La colonne 'date' doit être convertie en datetime."""
        datasets = {"clients": sample_clients, "touchpoints": sample_touchpoints}
        result = clean_datasets(datasets)
        assert pd.api.types.is_datetime64_any_dtype(result["touchpoints"]["date"]), (
            "La colonne 'date' doit être de type datetime"
        )

    def test_is_first_touch_converted_to_bool(self, sample_clients, sample_touchpoints):
        """La colonne 'is_first_touch' doit être convertie en bool."""
        datasets = {"clients": sample_clients, "touchpoints": sample_touchpoints}
        result = clean_datasets(datasets)
        assert result["touchpoints"]["is_first_touch"].dtype == bool, (
            "La colonne 'is_first_touch' doit être de type bool"
        )

    def test_is_last_touch_converted_to_bool(self, sample_clients, sample_touchpoints):
        """La colonne 'is_last_touch' doit être convertie en bool."""
        datasets = {"clients": sample_clients, "touchpoints": sample_touchpoints}
        result = clean_datasets(datasets)
        assert result["touchpoints"]["is_last_touch"].dtype == bool, (
            "La colonne 'is_last_touch' doit être de type bool"
        )

    def test_clients_unchanged(self, sample_clients, sample_touchpoints):
        """La table clients ne doit pas être modifiée."""
        datasets = {"clients": sample_clients, "touchpoints": sample_touchpoints}
        result = clean_datasets(datasets)
        pd.testing.assert_frame_equal(result["clients"], sample_clients)

    def test_no_rows_lost(self, sample_clients, sample_touchpoints):
        """Le nettoyage ne doit pas supprimer de lignes."""
        datasets = {"clients": sample_clients, "touchpoints": sample_touchpoints}
        result = clean_datasets(datasets)
        assert len(result["touchpoints"]) == len(sample_touchpoints), (
            "Le nettoyage ne doit pas supprimer de lignes"
        )


# ── Tests : compute_attribution ───────────────────────────────────────────────

class TestComputeAttribution:
    """Tests pour la fonction compute_attribution."""

    def test_returns_dataframe(self, sample_touchpoints):
        """La fonction doit retourner un DataFrame."""
        result = compute_attribution(sample_touchpoints)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_touchpoints):
        """Le DataFrame doit contenir les 5 colonnes attendues."""
        result = compute_attribution(sample_touchpoints)
        expected_cols = {"canal", "last_click", "first_click", "linear", "time_decay"}
        assert set(result.columns) == expected_cols

    def test_last_click_sums_to_100(self, sample_touchpoints):
        """La somme des last_click doit être égale à 100%."""
        result = compute_attribution(sample_touchpoints)
        assert abs(result["last_click"].sum() - 100) < 0.1, (
            "La somme des last_click doit être ~100%"
        )

    def test_first_click_sums_to_100(self, sample_touchpoints):
        """La somme des first_click doit être égale à 100%."""
        result = compute_attribution(sample_touchpoints)
        assert abs(result["first_click"].sum() - 100) < 0.1

    def test_linear_sums_to_100(self, sample_touchpoints):
        """La somme du modèle linear doit être égale à 100%."""
        result = compute_attribution(sample_touchpoints)
        assert abs(result["linear"].sum() - 100) < 0.1

    def test_time_decay_sums_to_100(self, sample_touchpoints):
        """La somme du modèle time_decay doit être égale à 100%."""
        result = compute_attribution(sample_touchpoints)
        assert abs(result["time_decay"].sum() - 100) < 0.1

    def test_email_dominates_last_click(self, sample_touchpoints):
        """
        Email doit avoir le plus grand crédit en last_click :
        les 2 clients convertis ont Email comme dernier canal.
        """
        result = compute_attribution(sample_touchpoints)
        top_canal = result.sort_values("last_click", ascending=False).iloc[0]["canal"]
        assert top_canal == "Email", (
            f"Email doit dominer en last_click, got '{top_canal}'"
        )

    def test_only_converted_clients_counted(self, sample_touchpoints):
        """
        Facebook ne doit pas apparaître dans l'attribution :
        le seul client Facebook (client 3) n'a pas converti.
        """
        result = compute_attribution(sample_touchpoints)
        assert "Facebook" not in result["canal"].values, (
            "Facebook ne doit pas apparaître — client 3 non converti"
        )


# ── Tests : compute_canal_stats ───────────────────────────────────────────────

class TestComputeCanalStats:
    """Tests pour la fonction compute_canal_stats."""

    def test_returns_dataframe(self, sample_clients, sample_touchpoints):
        """La fonction doit retourner un DataFrame."""
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_clients, sample_touchpoints):
        """Le DataFrame doit contenir les colonnes attendues."""
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        expected_cols = {
            "canal", "nb_last_touch", "nb_conversions",
            "n_touches_moyen", "taux_conversion"
        }
        assert expected_cols.issubset(set(result.columns))

    def test_taux_conversion_between_0_and_100(self, sample_clients, sample_touchpoints):
        """Le taux de conversion doit être entre 0 et 100."""
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        assert result["taux_conversion"].between(0, 100).all(), (
            "Tous les taux de conversion doivent être entre 0 et 100"
        )

    def test_email_conversion_rate_is_100(self, sample_clients, sample_touchpoints):
        """
        Email doit avoir un taux de conversion de 100% :
        il est last_touch pour les 2 clients convertis sur 2 last_touch Email.
        """
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        email_rate = result[result["canal"] == "Email"]["taux_conversion"].values[0]
        assert email_rate == 100.0, (
            f"Email doit avoir 100% de conversion, got {email_rate}"
        )

    def test_facebook_conversion_rate_is_0(self, sample_clients, sample_touchpoints):
        """
        Facebook doit avoir un taux de conversion de 0% :
        le seul client Facebook (client 3) n'a pas converti.
        """
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        fb_rate = result[result["canal"] == "Facebook"]["taux_conversion"].values[0]
        assert fb_rate == 0.0, (
            f"Facebook doit avoir 0% de conversion, got {fb_rate}"
        )

    def test_nb_conversions_never_exceeds_nb_last_touch(self, sample_clients, sample_touchpoints):
        """nb_conversions ne peut pas dépasser nb_last_touch."""
        result = compute_canal_stats(sample_touchpoints, sample_clients)
        assert (result["nb_conversions"] <= result["nb_last_touch"]).all(), (
            "nb_conversions ne peut pas dépasser nb_last_touch"
        )
