"""
test_generate.py
----------------
Tests unitaires pour src/generate.py.

Fonctions testées :
    - _sigmoid                  : transformation score → probabilité
    - _compute_conversion_proba : calcul de la probabilité de conversion

Lancer avec : pytest tests/ -v
"""

import pytest
from src.generate import _sigmoid, _compute_conversion_proba


# ── Tests : _sigmoid ──────────────────────────────────────────────────────────

class TestSigmoid:
    """Tests pour la fonction _sigmoid."""

    def test_zero_returns_half(self):
        """sigmoid(0) doit retourner exactement 0.5."""
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_output_between_0_and_1(self):
        """La sortie doit toujours être entre 0 et 1."""
        for x in [-10, -1, 0, 1, 10]:
            result = _sigmoid(x)
            assert 0 < result < 1, f"sigmoid({x}) = {result} hors de ]0, 1["

    def test_positive_input_above_half(self):
        """Un score positif doit donner une probabilité > 0.5."""
        assert _sigmoid(1.0) > 0.5
        assert _sigmoid(5.0) > 0.5

    def test_negative_input_below_half(self):
        """Un score négatif doit donner une probabilité < 0.5."""
        assert _sigmoid(-1.0) < 0.5
        assert _sigmoid(-5.0) < 0.5

    def test_monotonically_increasing(self):
        """sigmoid doit être strictement croissante."""
        values = [-3, -1, 0, 1, 3]
        results = [_sigmoid(x) for x in values]
        assert results == sorted(results), "sigmoid doit être strictement croissante"

    def test_large_positive_approaches_1(self):
        """Un score très élevé doit approcher 1."""
        assert _sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_0(self):
        """Un score très négatif doit approcher 0."""
        assert _sigmoid(-100) == pytest.approx(0.0, abs=1e-6)


# ── Tests : _compute_conversion_proba ─────────────────────────────────────────

class TestComputeConversionProba:
    """Tests pour la fonction _compute_conversion_proba."""

    # Paramètres de base réutilisés dans plusieurs tests
    BASE_PARAMS = {
        "canal_last":      "Email",
        "segment":         "Standard",
        "age":             35,
        "anciennete_mois": 12,
        "n_touches":       3,
        "canaux_parcours": ["SEO", "Email"],
    }

    def test_output_between_0_and_1(self):
        """La probabilité retournée doit être entre 0 et 1."""
        proba = _compute_conversion_proba(**self.BASE_PARAMS)
        assert 0 <= proba <= 1, f"Probabilité hors de [0, 1] : {proba}"

    def test_premium_converts_better_than_churner(self):
        """
        Un client Premium doit avoir une probabilité de conversion
        plus élevée qu'un client Churner, toutes choses égales par ailleurs.
        """
        import numpy as np
        np.random.seed(0)  # fixe le bruit pour comparaison équitable

        proba_premium = _compute_conversion_proba(
            **{**self.BASE_PARAMS, "segment": "Premium"}
        )
        np.random.seed(0)
        proba_churner = _compute_conversion_proba(
            **{**self.BASE_PARAMS, "segment": "Churner"}
        )
        assert proba_premium > proba_churner, (
            f"Premium ({proba_premium:.3f}) doit convertir mieux que Churner ({proba_churner:.3f})"
        )

    def test_email_converts_better_than_youtube(self):
        """
        Email (meilleur canal last-touch) doit donner une probabilité
        plus élevée que YouTube (moins bon canal).
        """
        import numpy as np
        np.random.seed(0)
        proba_email = _compute_conversion_proba(**self.BASE_PARAMS)

        np.random.seed(0)
        proba_youtube = _compute_conversion_proba(
            **{**self.BASE_PARAMS, "canal_last": "YouTube",
               "canaux_parcours": ["SEO", "YouTube"]}
        )
        assert proba_email > proba_youtube, (
            f"Email ({proba_email:.3f}) doit convertir mieux que YouTube ({proba_youtube:.3f})"
        )

    def test_more_touches_increases_proba(self):
        """
        Un parcours avec plus de touchpoints doit avoir une probabilité
        de conversion plus élevée (signal n_touches_bonus).
        """
        import numpy as np
        np.random.seed(0)
        proba_1_touch = _compute_conversion_proba(
            **{**self.BASE_PARAMS, "n_touches": 1,
               "canaux_parcours": ["Email"]}
        )
        np.random.seed(0)
        proba_5_touches = _compute_conversion_proba(
            **{**self.BASE_PARAMS, "n_touches": 5,
               "canaux_parcours": ["SEO", "Google Ads", "Instagram", "Facebook", "Email"]}
        )
        assert proba_5_touches > proba_1_touch, (
            f"5 touches ({proba_5_touches:.3f}) doit convertir mieux que 1 touch ({proba_1_touch:.3f})"
        )

    def test_discovery_to_closing_sequence_bonus(self):
        """
        Un parcours SEO → Email (discovery → closing) doit convertir
        mieux qu'un parcours Facebook → Instagram (sans bonus de séquence).
        """
        import numpy as np
        np.random.seed(0)
        proba_ideal = _compute_conversion_proba(
            **{**self.BASE_PARAMS,
               "canal_last": "Email",
               "canaux_parcours": ["SEO", "Email"]}
        )
        np.random.seed(0)
        proba_no_bonus = _compute_conversion_proba(
            **{**self.BASE_PARAMS,
               "canal_last": "Instagram",
               "canaux_parcours": ["Facebook", "Instagram"]}
        )
        assert proba_ideal > proba_no_bonus, (
            f"Séquence idéale ({proba_ideal:.3f}) doit convertir mieux "
            f"que séquence sans bonus ({proba_no_bonus:.3f})"
        )

    def test_all_canaux_accepted(self):
        """La fonction doit accepter tous les canaux définis sans erreur."""
        canaux = ["Email", "SEO", "Google Ads", "Instagram", "Facebook", "YouTube"]
        for canal in canaux:
            proba = _compute_conversion_proba(
                canal_last=canal,
                segment="Standard",
                age=35,
                anciennete_mois=12,
                n_touches=1,
                canaux_parcours=[canal],
            )
            assert 0 <= proba <= 1, f"Probabilité invalide pour canal '{canal}'"
