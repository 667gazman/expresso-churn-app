"""
Application Streamlit pour prédire le churn des clients Expresso.
Architecture : 4 pages (Contexte, Exploration, Modélisation, Prédiction)
"""

from pandas_profiling import ProfileReport
import streamlit as st
import pandas as pd
import logging
import sys
import os
from ydata_profiling import ProfileReport   # ou data_profiling si tu as migré
from streamlit_pandas_profiling import st_profile_report

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import prepare_data
from modeling import load_model, train_model
from utils import get_target, load_data
from inference import predict

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Expresso Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# INITIALISER LA SESSION STREAMLIT
# ============================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data_raw = None
    st.session_state.data_prepared = None
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.model = None
    st.session_state.model_trained = False


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🔧 Configuration")

# Chemin du dataset
data_path = st.sidebar.text_input(
    "Chemin vers le CSV",
    value="data/expresso_churn.csv"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Actions rapides")

# Bouton : Charger données
if st.sidebar.button("🔄 Charger les données"):
    with st.spinner("Chargement..."):
        df, success, error = load_data(data_path)
        if success:
            st.session_state.data_raw = df
            st.session_state.data_loaded = True
            st.sidebar.success(f"✓ {df.shape[0]} lignes chargées")
        else:
            st.sidebar.error(f"❌ {error}")

# Bouton : Préparer les données
if st.sidebar.button("🧹 Préparer les données"):
    if not st.session_state.data_loaded:
        st.sidebar.error("⚠️ Chargez les données d'abord")
    else:
        with st.spinner("Préparation..."):
            try:
                df_prepared = prepare_data(st.session_state.data_raw)
                st.session_state.data_prepared = df_prepared
                
                # Séparer X et y
                X, y, success, error = get_target(df_prepared, target_col='CHURN')
                if success:
                    st.session_state.X = X
                    st.session_state.y = y
                    st.sidebar.success("✓ Données préparées")
                else:
                    st.sidebar.error(f"❌ {error}")
            except Exception as e:
                st.sidebar.error(f"❌ Erreur: {str(e)}")

# Bouton : Entraîner le modèle
if st.sidebar.button("🤖 Entraîner le modèle"):
    if st.session_state.X is None:
        st.sidebar.error("⚠️ Préparez les données d'abord")
    else:
        with st.spinner("Entraînement en cours..."):
            model, accuracy, success, error = train_model(
                st.session_state.X,
                st.session_state.y
            )
            if success:
                st.session_state.model = model
                st.session_state.model_trained = True
                st.sidebar.success(f"✓ Modèle entraîné (Acc: {accuracy:.2%})")
            else:
                st.sidebar.error(f"❌ {error}")

# Charger automatiquement le modèle existant
if not st.session_state.model_trained:
    model, success, error = load_model()
    if success:
        st.session_state.model = model
        st.session_state.model_trained = True

# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📑 Pages")

page = st.sidebar.radio(
    "Sélectionner une page",
    ["Contexte", "Exploration", "Modélisation", "Prédiction"]
)

# ============================================================
# PAGE 1 : CONTEXTE
# ============================================================
if page == "Contexte":
    st.title("📋 Contexte du Projet - Expresso Churn")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objectif
        Prédire les clients à risque de résilier leur abonnement télécoms.
        
        ### 📊 Dataset
        - **Source** : Données clients Expresso Telecom
        - **Format** : CSV avec variables démographiques et comportementales
        - **Cible** : Variable binaire CHURN (0 = reste, 1 = quitte)
        
        ### 🚀 Pipeline ML
        1. **Chargement** : CSV → DataFrame Pandas
        2. **Nettoyage** : Gestion des valeurs manquantes
        3. **Encodage** : One-Hot pour variables catégorielles (REGION)
        4. **Modélisation** : RandomForest
        5. **Prédiction** : Probabilités de churn
        
        ### ✅ Utilisation
        1. Cliquez "🔄 Charger les données" (sidebar)
        2. Cliquez "🧹 Préparer les données"
        3. Cliquez "🤖 Entraîner le modèle"
        4. Allez à "Prédiction" pour tester
        """)
    
    with col2:
        st.info(f"""
        ### 📈 Statut
        
        **Données chargées**
        {'✅ Oui' if st.session_state.data_loaded else '❌ Non'}
        
        **Données préparées**
        {'✅ Oui' if st.session_state.data_prepared is not None else '❌ Non'}
        
        **Modèle entraîné**
        {'✅ Oui' if st.session_state.model_trained else '❌ Non'}
        """)


# ============================================================
# PAGE 2 : EXPLORATION
# ============================================================
elif page == "Exploration":
    st.title("📊 Exploration des données")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Chargez les données d'abord (sidebar → 🔄 Charger les données)")
    else:
        df = st.session_state.data_raw
        
        # Statistiques générales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lignes", f"{df.shape[0]:,}")
        with col2:
            st.metric("Colonnes", df.shape[1])
        with col3:
            st.metric("Colonnes numériques", len(df.select_dtypes(include=['number']).columns))
        with col4:
            st.metric("Colonnes texte", len(df.select_dtypes(exclude=['number']).columns))
        
        st.markdown("---")
        
        # Aperçu des données
        st.subheader("👁️ Aperçu")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Info colonnes
        st.subheader("📋 Info Colonnes")
        info_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null': df.notna().sum(),
            'Manquantes': df.isnull().sum(),
            'Uniques': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True)
        
        st.markdown("---")
        
        # Statistiques
        st.subheader("📈 Statistiques")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Profiling pandas
        st.subheader("🔎 Profiling du dataset")
        from ydata_profiling import ProfileReport
        from streamlit_pandas_profiling import st_profile_report
        
        profile = ProfileReport(df, title="Rapport Expresso Churn", explorative=True)
        st_profile_report(profile)

# ============================================================
# PAGE 4 : MODÉLISATION
# ============================================================
elif page == "Modélisation":
    st.title("🤖 Modélisation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Pipeline")
        
        # Statut du pipeline
        pipeline_items = [
            ("Données chargées", st.session_state.data_loaded),
            ("Données préparées", st.session_state.data_prepared is not None),
            ("X et y disponibles", st.session_state.X is not None),
            ("Modèle entraîné", st.session_state.model_trained)
        ]
        
        for item, status in pipeline_items:
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} {item}")
        
        st.markdown("---")
        
        # Afficher les métadonnées du modèle
        if st.session_state.model_trained:
            st.subheader("🎯 Modèle")
            st.markdown(f"""
            - **Type** : RandomForest
            - **n_estimators** : 100
            - **max_depth** : 15
            - **État** : Entraîné et sauvegardé
            """)
    
    with col2:
        st.subheader("📈 Résultats d'entraînement")
        
        if st.session_state.model_trained:
            st.success("✓ Modèle disponible")
            
            # Afficher les infos du modèle
            model = st.session_state.model
            if hasattr(model, 'n_features_in_'):
                st.markdown(f"**Nombre de features** : {model.n_features_in_}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.markdown("**Top 10 Features par importance**")
                
                # Créer un DataFrame avec les importances
                feature_imp = pd.DataFrame({
                    'Feature': st.session_state.X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(feature_imp.set_index('Feature')['Importance'])
        else:
            st.info("Entraîner un modèle d'abord via la sidebar")


# ============================================================
# PAGE 5 : PRÉDICTION
# ============================================================
elif page == "Prédiction":
    st.title("🔮 Prédiction du Churn")
    
    if not st.session_state.model_trained:
        st.error("❌ Aucun modèle disponible. Entraîner d'abord via la sidebar.")
    else:
        st.success("✓ Modèle chargé et prêt")
        
        mode = st.radio(
            "Mode de prédiction",
            ["Prédiction unique", "Prédiction en batch"],
            horizontal=True
        )
        
        # ======== MODE 1 : PRÉDICTION UNIQUE ========
        if mode == "Prédiction unique":
            st.subheader("📝 Entrez les valeurs pour un client")
            
            if st.session_state.data_prepared is None:
                st.warning("⚠️ Préparez les données d'abord")
            else:
                # Récupérer les colonnes features
                feature_cols = st.session_state.X.columns.tolist()
                
                # Créer des inputs
                input_dict = {}
                cols = st.columns(3)
                
                for i, col_name in enumerate(feature_cols):
                    with cols[i % 3]:
                        # Vérifier le type de la colonne
                        if col_name in st.session_state.data_raw.columns:
                            col_type = st.session_state.data_raw[col_name].dtype
                            
                            if col_type in ['float64', 'int64']:
                                # Input numérique
                                default_val = float(st.session_state.data_raw[col_name].median())
                                input_dict[col_name] = st.number_input(
                                    col_name,
                                    value=default_val,
                                    format="%.2f"
                                )
                            else:
                                # Input catégoriel
                                try:
                                    options = st.session_state.data_raw[col_name].unique()[:10].tolist()
                                    input_dict[col_name] = st.selectbox(col_name, options)
                                except:
                                    input_dict[col_name] = st.text_input(col_name)
                        else:
                            # Colonne encodée (créée par one-hot)
                            input_dict[col_name] = st.number_input(col_name, value=0.0)
                
                if st.button("🔮 Prédire", key="single_predict"):
                    # Créer un DataFrame avec les inputs
                    input_df = pd.DataFrame([input_dict])
                    
                    # Adapter la structure si nécessaire
                    for col in feature_cols:
                        if col not in input_df.columns:
                            input_df[col] = 0.0
                    
                    input_df = input_df[feature_cols]
                    
                    # Prédiction
                    predictions, probabilities, success, error = predict(
                        st.session_state.model,
                        input_df
                    )
                    
                    if success:
                        pred = predictions[0]
                        prob = probabilities[0]
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pred_label = "🔴 CHURN" if pred == 1 else "🟢 RETAIN"
                            st.metric("Prédiction", pred_label)
                        
                        with col2:
                            st.metric("Probabilité", f"{prob:.1%}")
                        
                        with col3:
                            if prob >= 0.7:
                                risk = "🔴 ÉLEVÉ"
                            elif prob >= 0.4:
                                risk = "🟡 MOYEN"
                            else:
                                risk = "🟢 FAIBLE"
                            st.metric("Risque", risk)
                    else:
                        st.error(f"❌ Erreur: {error}")
        
        # ======== MODE 2 : PRÉDICTION EN BATCH ========
        else:
            st.subheader("📁 Prédiction en batch")
            
            uploaded_file = st.file_uploader("Téléchargez un CSV", type=['csv'])
            
            if uploaded_file:
                df_batch = pd.read_csv(uploaded_file)
                st.write(f"Fichier chargé: {df_batch.shape[0]} lignes, {df_batch.shape[1]} colonnes")
                
                st.dataframe(df_batch.head(5), use_container_width=True)
                
                if st.button("🔮 Prédire sur le batch"):
                    # Préparer les données du batch
                    try:
                        df_batch_prepared = prepare_data(df_batch)
                        
                        # Séparer X et y (s'il existe)
                        if 'CHURN' in df_batch_prepared.columns:
                            X_batch = df_batch_prepared.drop('CHURN', axis=1)
                        else:
                            X_batch = df_batch_prepared
                        
                        # Prédictions
                        predictions, probabilities, success, error = predict(
                            st.session_state.model,
                            X_batch
                        )
                        
                        if success:
                            # Créer un résultat DataFrame
                            result_df = df_batch.copy()
                            result_df['PREDICTION'] = ['CHURN' if p == 1 else 'RETAIN' for p in predictions]
                            result_df['PROBABILITY'] = probabilities
                            result_df['PROBABILITY_PCT'] = (probabilities * 100).round(2)
                            
                            st.success(f"✓ {len(result_df)} prédictions effectuées")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Télécharger les résultats
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger les résultats (CSV)",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"❌ Erreur: {error}")
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la préparation du batch: {str(e)}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    <p>Expresso Churn Prediction | Pipeline ML Robuste</p>
</div>
""", unsafe_allow_html=True)