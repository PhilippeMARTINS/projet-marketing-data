"""
model.py
--------
Module ML : prédiction de conversion client à partir du parcours multitouch.
Modèle : Random Forest Classifier (Scikit-learn)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
import joblib
from src.load import query_sqlite


OUTPUT_PATH = Path("outputs")
MODEL_PATH  = Path("data/processed/model.pkl")


def build_features() -> pd.DataFrame:
    """
    Construit la table de features par client pour le modèle ML.

    Features utilisées :
    - Profil client : age, segment, region, ancienneté
    - Parcours : nombre de touchpoints, canaux utilisés (one-hot),
                 canal first-touch, canal last-touch

    Returns:
        pd.DataFrame: Table de features prête pour Sklearn
    """
    sql = """
        SELECT
            c.client_id,
            c.age,
            c.segment,
            c.region,
            c.anciennete_mois,
            t_agg.n_touches,
            t_agg.first_canal,
            t_agg.last_canal,
            t_agg.converti
        FROM clients c
        JOIN (
            SELECT
                client_id,
                MAX(n_touches_total)                          AS n_touches,
                MAX(CASE WHEN is_first_touch = 1 THEN canal END) AS first_canal,
                MAX(CASE WHEN is_last_touch  = 1 THEN canal END) AS last_canal,
                MAX(converti)                                 AS converti
            FROM touchpoints
            GROUP BY client_id
        ) t_agg ON c.client_id = t_agg.client_id
    """
    df = query_sqlite(sql)
    print(f"✅ Features construites — {df.shape[0]} clients, {df.shape[1]} colonnes")
    return df


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Encode les variables catégorielles et prépare X, y.

    Returns:
        tuple: (X, y, feature_names)
    """
    df = df.copy()

    # Encodage des variables catégorielles
    cat_cols = ["segment", "region", "first_canal", "last_canal"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = ["age", "segment", "region", "anciennete_mois",
                    "n_touches", "first_canal", "last_canal"]

    X = df[feature_cols]
    y = df["converti"]

    print(f"✅ Preprocessing terminé — {X.shape[1]} features")
    print(f"   Répartition cible : {y.value_counts().to_dict()}")
    return X, y, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Entraîne un Random Forest et évalue ses performances.

    Returns:
        tuple: (modèle entraîné, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Évaluation
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Performances du modèle :")
    print(f"   AUC-ROC : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return model, X_test, y_test, y_proba


def plot_feature_importance(model, feature_names: list) -> None:
    """Graphique d'importance des features."""
    importances = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importances, x="importance", y="feature",
                hue="feature", palette="Blues_r", legend=False, ax=ax)
    ax.set_title("Importance des features — Random Forest",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "feature_importance.png", dpi=150)
    plt.close()
    print("✅ feature_importance.png sauvegardé")


def plot_roc_curve(y_test, y_proba) -> None:
    """Courbe ROC du modèle."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", linewidth=2,
            label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--",
            linewidth=1, label="Aléatoire (AUC = 0.50)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC — Prédiction de conversion",
                 fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "roc_curve.png", dpi=150)
    plt.close()
    print("✅ roc_curve.png sauvegardé")


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non converti", "Converti"],
                yticklabels=["Non converti", "Converti"])
    ax.set_title("Matrice de confusion", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "confusion_matrix.png", dpi=150)
    plt.close()
    print("✅ confusion_matrix.png sauvegardé")


def save_model(model) -> None:
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé : {MODEL_PATH}")

def plot_shap(model, X_test: pd.DataFrame, feature_names: list) -> None:
    """
    Génère les visualisations SHAP pour expliquer les prédictions du modèle.
    - Summary plot : impact global de chaque feature (barres)
    - Waterfall plot : explication d'une prédiction individuelle
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")

    print("⏳ Calcul des valeurs SHAP...")

    # Utilisation de l'API moderne Explanation
    explainer = shap.TreeExplainer(model)
    shap_exp  = explainer(X_test)

    # Classe 1 uniquement (convertis)
    shap_exp_class1 = shap_exp[:, :, 1]

    # ── Summary plot (barres) ─────────────────────────────────────────────
    shap.summary_plot(
        shap_exp_class1.values,
        X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP — Impact des features sur la conversion",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("✅ shap_summary.png sauvegardé")

    # ── Waterfall plot (1er client) ───────────────────────────────────────
    shap.plots.waterfall(shap_exp_class1[0], show=False)
    plt.title("SHAP — Explication d'une prédiction individuelle",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("✅ shap_waterfall.png sauvegardé")

def run_model() -> None:
    """Point d'entrée principal du module ML."""
    print("=== MODÈLE ML ===")
    df           = build_features()
    X, y, feats  = preprocess(df)
    model, X_test, y_test, y_proba = train_model(X, y)
    y_pred       = model.predict(X_test)

    plot_feature_importance(model, feats)
    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_shap(model, X_test, feats)
    save_model(model)
    print("\n✅ Module ML terminé")


if __name__ == "__main__":
    run_model()