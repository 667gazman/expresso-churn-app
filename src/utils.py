"""
Module utilitaire pour le projet Expresso Churn.
Contient les fonctions d'utilitaires et d'accès aux données.
"""

import pandas as pd
import logging

# ============================================================
# CONFIGURATION DU LOGGER
# ============================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_target(df, target_col="CHURN"):
    """
    Sépare les features (X) et la cible (y) d'un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame complet
        target_col (str): Nom de la colonne cible
        
    Returns:
        tuple: (X, y, success: bool, error_msg: str)
            - X: DataFrame des features
            - y: Series de la cible
    """
    try:
        if target_col not in df.columns:
            error_msg = f"La colonne '{target_col}' est introuvable. Colonnes disponibles : {df.columns.tolist()}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        logger.info(f"Séparation X/y: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Distribution de y: {y.value_counts().to_dict()}")
        
        return X, y, True, ""
    
    except KeyError as e:
        error_msg = f"Erreur lors de la séparation X/y: {str(e)}"
        logger.error(error_msg)
        return None, None, False, error_msg
    
    except Exception as e:
        error_msg = f"Erreur inattendue lors de la séparation X/y: {str(e)}"
        logger.error(error_msg)
        return None, None, False, error_msg


def load_data(filepath):
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        filepath (str): Chemin vers le fichier CSV
        
    Returns:
        tuple: (df, success: bool, error_msg: str)
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        logger.info(f"Colonnes: {list(df.columns)}")
        return df, True, ""
    
    except FileNotFoundError:
        error_msg = f"Fichier non trouvé: {filepath}"
        logger.error(error_msg)
        return None, False, error_msg
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement: {str(e)}"
        logger.error(error_msg)
        return None, False, error_msg