import streamlit as st
import pandas as pd
from src.preprocessing import prepare_data
from src.modeling import load_model, train_model, save_model
from src.utils import get_target
from src.inference import predict

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["Contexte", "Exploration", "Modélisation", "Prédiction"])


@st.cache_data
def load_data():
    df = pd.read_csv("data/expresso_churn.csv")
    return df


# ========================
# PAGE 1
# ========================
if page == "Contexte":
    st.title("Expresso Churn")
    st.write("Prédiction du churn client dans les télécoms.")

# ========================
# PAGE 2
# ========================
elif page == "Exploration":
    df = load_data()
    st.dataframe(df.head())

# ========================
# PAGE 3
# ========================
elif page == "Modélisation":
    df = load_data()
    df = prepare_data(df)

    X, y = get_target(df)

    if st.button("Entraîner le modèle"):
        model, score = train_model(X, y)
        save_model(model)
        st.success(f"Modèle entraîné ! Accuracy : {score:.2f}")

# ========================
# PAGE 4 🔥
# ========================
elif page == "Prédiction":
    st.subheader("Faire une prédiction")

    model = load_model()

    # Inputs utilisateur
    tenure = st.number_input("TENURE", 0)
    montant = st.number_input("MONTANT", 0)

    input_df = pd.DataFrame([{
        "TENURE": tenure,
        "MONTANT": montant
    }])

    input_df = prepare_data(input_df)

    if st.button("Prédire"):
        pred, proba = predict(model, input_df)
        st.write(f"Churn: {pred[0]} | Probabilité: {proba[0]:.2f}")