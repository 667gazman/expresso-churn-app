# Expresso Churn Prediction

## Description du jeu de données
Expresso est une entreprise africaine de services de télécommunications qui fournit des services de télécommunications sur deux marchés africains : la Mauritanie et le Sénégal.  
Les données décrivent 2,5 millions de clients Expresso avec plus de 15 variables de comportement afin de prédire la probabilité de désabonnement des clients.

---

## Problèmes rencontrés et solutions

- **Problème : Calcul de médiane sur colonnes texte**  
  - Le preprocessing appliquait `.median()` sur des colonnes catégorielles comme `"K > 24 month"`, ce qui provoquait `TypeError: Cannot convert [...] to numeric`.  
  - **Solutions :**  
    - Supprimer l’ancienne version de `clean_data` qui traitait toutes les colonnes de la même façon.  
    - Ajouter une condition `pd.api.types.is_numeric_dtype(df[col])` pour n’appliquer la médiane que sur les colonnes numériques.  
    - Utiliser `.mode()[0]` pour remplir les colonnes catégorielles.

- **Problème : Colonnes catégorielles non encodées avant l’entraînement**  
  - Le modèle RandomForest ne pouvait pas convertir des chaînes de caractères en float (`could not convert string to float`).  
  avec l'erreur :
  TypeError: Cannot convert ['K > 24 month' 'I 18-21 month' 'K > 24 month' ... 'K > 24 month'
 'K > 24 month' 'K > 24 month'] to numeric

File "C:\Users\EliteBook 755 G5\Desktop\python_new\myenv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "C:\Users\EliteBook 755 G5\Desktop\python_new\myenv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - **Solutions :**  
    - Encodage automatique de toutes les colonnes catégorielles avec `pd.get_dummies`.  
    - Vérifier que seules des colonnes numériques sont envoyées au modèle.

- **Problème : Profilage pandas mal intégré**  
  

- **Problème : Modèle non trouvé lors de la prédiction**  
  - Si `model.pkl` n’était pas encore créé, la page Prédiction plantait.  
  - **Solutions :**  
    - Ajouter un `try/except FileNotFoundError` dans `load_model()`.  
    - Afficher un message clair : "⚠️ Aucun modèle trouvé. Entraînez-le d'abord."

- **Problème : Formulaire de prédiction trop limité**  
  - Initialement, seuls `TENURE` et `MONTANT` étaient utilisés pour la prédiction.  
  - **Solutions :**  
    - Générer dynamiquement les champs dans Streamlit :  
      - `number_input` pour colonnes numériques.  
      - `selectbox` pour colonnes catégorielles.  
    - Assurer que la structure des inputs correspond aux features du modèle.

---

## Conclusion
Ce projet m’a permis de construire un pipeline ML complet (prétraitement, modélisation, prédiction, interface Streamlit) et de résoudre plusieurs problèmes liés au traitement des données et au déploiement. Les solutions mises en place garantissent un workflow plus robuste et reproductible.
