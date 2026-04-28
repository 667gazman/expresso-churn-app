def predict(model, input_df):
    proba = model.predict_proba(input_df)[:, 1]
    prediction = (proba > 0.5).astype(int)
    return prediction, proba