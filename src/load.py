"""
load.py
-------
Module de chargement : sauvegarde des tables dans SQLite.
"""

import sqlite3
import pandas as pd
from pathlib import Path


DB_PATH = Path("data/processed/marketing.db")


def save_to_sqlite(data: dict) -> None:
    """
    Sauvegarde toutes les tables dans la base SQLite.

    Args:
        data: dict contenant clients, touchpoints, attribution, canal_stats
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    tables = {
        "clients":     data["clients"],
        "touchpoints": data["touchpoints"],
        "attribution": data["attribution"],
        "canal_stats": data["canal_stats"],
    }

    for table_name, df in tables.items():
        df_save = df.copy()

        # Conversion des colonnes non supportées par SQLite
        for col in df_save.select_dtypes(include=["datetime64[ns]"]).columns:
            df_save[col] = df_save[col].astype(str)
        for col in df_save.select_dtypes(include=["bool"]).columns:
            df_save[col] = df_save[col].astype(int)

        df_save.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"✅ '{table_name}' sauvegardé — {len(df_save)} lignes")

    conn.close()
    print(f"\n✅ Base SQLite prête : {DB_PATH}")


def query_sqlite(sql: str) -> pd.DataFrame:
    """
    Exécute une requête SQL sur la base et retourne un DataFrame.

    Args:
        sql: Requête SQL à exécuter

    Returns:
        pd.DataFrame: Résultat de la requête
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df