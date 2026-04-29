"""
Module d'entraînement et chargement de modèles pour le projet Expresso Churn.
"""

import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

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

MODEL_PATH = "models/model.pkl"


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Entraîne un modèle RandomForest et le sauvegarde.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Cible
        test_size (float): Proportion de test
        random_state (int): Seed pour reproductibilité
        
    Returns:
        tuple: (model, accuracy_score, success: bool, error_msg: str)
    """
    try:
        logger.info("=" * 70)
        logger.info("ENTRAÎNEMENT DU MODÈLE RANDOMFOREST")
        logger.info("=" * 70)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
        
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        logger.info("✓ Modèle entraîné")
        
        # Évaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Métriques supplémentaires
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        logger.info(f"  Train Accuracy: {model.score(X_train, y_train):.4f}")
        logger.info(f"  Test Accuracy: {accuracy:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Créer le répertoire models s'il n'existe pas
        os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else ".", exist_ok=True)
        
        # Sauvegarder le modèle
        joblib.dump(model, MODEL_PATH)
        logger.info(f"✓ Modèle sauvegardé: {MODEL_PATH}")
        
        logger.info("=" * 70)
        
        return model, accuracy, True, ""
    
    except Exception as e:
        error_msg = f"Erreur lors de l'entraînement: {str(e)}"
        logger.error(error_msg)
        return None, 0.0, False, error_msg


def load_model():
    """
    Charge le modèle sauvegardé si disponible.
    
    Returns:
        tuple: (model, success: bool, error_msg: str)
    """
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✓ Modèle chargé: {MODEL_PATH}")
            return model, True, ""
        else:
            error_msg = f"Le modèle n'existe pas encore. Lance un entraînement d'abord."
            logger.warning(error_msg)
            return None, False, error_msg
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement du modèle: {str(e)}"
        logger.error(error_msg)
        return None, False, error_msg