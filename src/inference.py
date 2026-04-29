"""
Module d'inférence pour le projet Expresso Churn.
Contient les fonctions de prédiction.
"""

import numpy as np
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


def predict(model, input_df, threshold=0.5):
    """
    Effectue une prédiction avec le modèle.
    
    Args:
        model: Modèle RandomForest entraîné
        input_df (pd.DataFrame): DataFrame avec les features
        threshold (float): Seuil de décision (défaut: 0.5)
        
    Returns:
        tuple: (predictions, probabilities, success: bool, error_msg: str)
            - predictions: array de 0/1
            - probabilities: array de probabilités [0-1]
    """
    try:
        if model is None:
            return None, None, False, "Modèle non disponible"
        
        if input_df is None or input_df.shape[0] == 0:
            return None, None, False, "DataFrame d'entrée vide"
        
        # Probabilité de la classe positive (churn = 1)
        proba = model.predict_proba(input_df)[:, 1]
        
        # Seuil de décision
        prediction = (proba >= threshold).astype(int)
        
        logger.info(f"Prédictions effectuées: {input_df.shape[0]} échantillons")
        
        return prediction, proba, True, ""
    
    except Exception as e:
        error_msg = f"Erreur lors de la prédiction: {str(e)}"
        logger.error(error_msg)
        return None, None, False, error_msg