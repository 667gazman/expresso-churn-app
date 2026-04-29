"""
Module de nettoyage et préparation des données pour le projet Expresso Churn.
Gère correctement les colonnes numériques vs catégorielles.
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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données en gérant les valeurs manquantes intelligemment.
    
    - Colonnes numériques : imputation par la médiane
    - Colonnes catégorielles : imputation par le mode
    - Supprime les colonnes inutiles
    - Gère les cas spéciaux (activité, région, etc.)
    
    Args:
        df (pd.DataFrame): DataFrame brut
        
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    df = df.copy()
    
    logger.info(f"Nettoyage des données - Shape initiale: {df.shape}")
    logger.info(f"Colonnes manquantes:\n{df.isnull().sum()}")
    
    # ======== 1. SUPPRIMER LES COLONNES INUTILES ========
    cols_to_drop = ['ZONE1', 'ZONE2', 'user_id']
    dropped = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=dropped, inplace=True)
    logger.info(f"Suppression des colonnes inutiles: {dropped}")
    
    # ======== 2. CAS SPÉCIAUX : REMPLISSAGE PAR 0 AVANT IMPUTATION GÉNÉRALE ========
    # Variables d'activité qui doivent être 0 si manquantes (traitement prioritaire)
    activity_cols = ['DATA_VOLUME', 'FREQUENCE_RECH', 'MONTANT']
    for col in activity_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(0, inplace=True)
            logger.info(f"  {col} (activité): remplissage par 0")
    
    # ======== 3. IMPUTATION GÉNÉRALE DES VALEURS MANQUANTES ========
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Colonnes numériques : médiane
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            logger.info(f"  {col} (numérique): médiane = {median_value}")
        else:
            # Colonnes catégorielles : mode, sinon 'Unknown'
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                mode_value = mode_values[0]
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"  {col} (catégoriel): mode = '{mode_value}'")
            else:
                df[col].fillna("Unknown", inplace=True)
                logger.info(f"  {col} (catégoriel): pas de mode → 'Unknown'")
    
    logger.info(f"Données nettoyées - Shape finale: {df.shape}")
    return df


def encode_features(df: pd.DataFrame, target_col: str = "CHURN") -> pd.DataFrame:
    """
    Encode TOUTES les colonnes catégorielles restantes en numériques via One-Hot Encoding.
    
    Contrairement à l'ancienne version qui n'encodait que REGION, cette fonction
    détecte automatiquement toutes les colonnes de type object/category et les encode,
    ce qui évite l'erreur "could not convert string to float" lors de l'entraînement.
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes catégorielles
        target_col (str): Nom de la colonne cible à exclure de l'encodage (défaut: "CHURN")
        
    Returns:
        pd.DataFrame: DataFrame avec toutes les features encodées en numérique
    """
    df = df.copy()
    
    # Identifier toutes les colonnes catégorielles, en excluant la cible
    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if col != target_col
    ]
    
    if not categorical_cols:
        logger.info("Aucune colonne catégorielle à encoder.")
        return df
    
    logger.info(f"Encodage one-hot de {len(categorical_cols)} colonne(s) catégorielle(s):")
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        logger.info(f"  → {col} ({unique_vals} valeurs uniques)")
    
    # ✅ CORRECTION PRINCIPALE : encoder TOUTES les colonnes catégorielles
    # drop_first=True évite la multicolinéarité parfaite
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    logger.info(f"Shape après encodage: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame, target_col: str = "CHURN") -> pd.DataFrame:
    """
    Pipeline complet : nettoyage + encodage.
    
    Garantit qu'aucune colonne de type string ne subsiste avant l'entraînement
    (à l'exception de la colonne cible si elle est textuelle).
    
    Args:
        df (pd.DataFrame): DataFrame brut
        target_col (str): Nom de la colonne cible à préserver (défaut: "CHURN")
        
    Returns:
        pd.DataFrame: DataFrame entièrement numérique, prêt pour le modèle
        
    Raises:
        Exception: si des colonnes non-numériques subsistent après encodage
    """
    try:
        logger.info("=" * 70)
        logger.info("PIPELINE DE PRÉPARATION DES DONNÉES")
        logger.info("=" * 70)
        
        # Étape 1 : Nettoyage
        df_clean = clean_data(df)
        
        # Étape 2 : Encodage de toutes les colonnes catégorielles
        df_encoded = encode_features(df_clean, target_col=target_col)
        
        # ======== VÉRIFICATION FINALE ========
        # S'assurer qu'aucune colonne texte ne subsiste (hors cible)
        remaining_cat_cols = [
            col for col in df_encoded.select_dtypes(include=['object', 'category']).columns
            if col != target_col
        ]
        if remaining_cat_cols:
            raise ValueError(
                f"Des colonnes non-numériques subsistent après encodage : {remaining_cat_cols}. "
                f"Vérifiez encode_features()."
            )
        
        logger.info("✓ Vérification : toutes les features sont numériques.")
        logger.info("✓ Pipeline complété avec succès.")
        logger.info("=" * 70)
        
        return df_encoded
    
    except Exception as e:
        error_msg = f"Erreur lors de la préparation des données: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)