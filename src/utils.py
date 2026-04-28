def get_target(df, target_col="CHURN"):
    """Retourne X et y en fonction de la colonne cible."""
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y
    else:
        raise KeyError(f"La colonne '{target_col}' est introuvable. Colonnes disponibles : {df.columns.tolist()}")
