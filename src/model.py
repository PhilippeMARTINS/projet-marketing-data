"""
model.py
--------
Module ML : prédiction de conversion client à partir du parcours multitouch.
Modèle : XGBoost Classifier avec cross-validation 5-fold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
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
    - Nouvelles features métier :
        * duree_parcours_jours : durée entre le 1er et dernier touchpoint
        * ratio_canaux_payants  : proportion de canaux payants (Ads) dans le parcours
        * nb_canaux_distincts   : diversité des canaux touchés

    Returns:
        pd.DataFrame: Table de features prête pour Sklearn/XGBoost
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
            t_agg.converti,
            t_agg.duree_parcours_jours,
            t_agg.nb_canaux_distincts,
            t_agg.ratio_canaux_payants
        FROM clients c
        JOIN (
            SELECT
                client_id,
                MAX(n_touches_total)                              AS n_touches,
                MAX(CASE WHEN is_first_touch = 1 THEN canal END) AS first_canal,
                MAX(CASE WHEN is_last_touch  = 1 THEN canal END) AS last_canal,
                MAX(converti)                                     AS converti,
                -- Durée du parcours en jours
                CAST(
                    (JULIANDAY(MAX(date)) - JULIANDAY(MIN(date)))
                AS INTEGER)                                       AS duree_parcours_jours,
                -- Nombre de canaux distincts utilisés
                COUNT(DISTINCT canal)                             AS nb_canaux_distincts,
                -- Ratio canaux payants (Google Ads, Facebook, Instagram)
                ROUND(
                    SUM(CASE WHEN canal IN ('Google Ads', 'Facebook', 'Instagram')
                        THEN 1.0 ELSE 0.0 END) / COUNT(*), 3
                )                                                 AS ratio_canaux_payants
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

    feature_cols = [
        "age", "segment", "region", "anciennete_mois",
        "n_touches", "first_canal", "last_canal",
        # Nouvelles features métier
        "duree_parcours_jours", "nb_canaux_distincts", "ratio_canaux_payants",
    ]

    X = df[feature_cols]
    y = df["converti"]

    print(f"✅ Preprocessing terminé — {X.shape[1]} features")
    print(f"   Répartition cible : {y.value_counts().to_dict()}")
    return X, y, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Entraîne un XGBoost avec cross-validation 5-fold et évalue ses performances.

    Returns:
        tuple: (modèle entraîné, X_test, y_test, y_proba)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calcul du ratio déséquilibre de classes pour scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"   scale_pos_weight = {scale_pos_weight:.2f} (gestion déséquilibre)")

    model = LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    # Cross-validation 5-fold pour un score robuste
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\n📊 Cross-validation AUC (5-fold) :")
    print(f"   Scores : {[round(s, 4) for s in cv_scores]}")
    print(f"   Moyenne : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement final sur tout le train set
    model.fit(X_train, y_train)

    # Évaluation sur le test set
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Performances sur le test set :")
    print(f"   AUC-ROC : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    return model, X_test, y_test, y_proba


def plot_feature_importance(model, feature_names: list) -> None:
    """Graphique d'importance des features (gain XGBoost)."""
    importances = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=importances, x="importance", y="feature",
                hue="feature", palette="Blues_r", legend=False, ax=ax)
    ax.set_title("Importance des features — LightGBM (gain)",
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
            label=f"XGBoost (AUC = {auc:.4f})")
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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé → {MODEL_PATH}")


def run_model() -> None:
    """Orchestration complète : features → preprocessing → train → plots → save."""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    df              = build_features()
    X, y, feat_cols = preprocess(df)
    model, X_test, y_test, y_proba = train_model(X, y)

    y_pred = model.predict(X_test)
    plot_feature_importance(model, feat_cols)
    plot_roc_curve(y_test, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    save_model(model)