def get_target(df, target_col="CHURN"):
    """Retourne X et y en fonction de la colonne cible."""
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y
    else:
        raise KeyError(f"La colonne '{target_col}' est introuvable. Colonnes disponibles : {df.columns.tolist()}")





"""
Module d'initialisation du package src
Contient les imports principaux et configuration
"""

from src.preprocessing import clean_data, encode_features, prepare_data
from src.modeling import train_model, load_model
from src.inference import predict
from src.utils import get_target

__all__ = [
    'clean_data',
    'encode_features', 
    'prepare_data',
    'train_model',
    'load_model',
    'predict',
    'get_target'
]