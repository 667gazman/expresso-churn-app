import pandas as pd

def clean_data(df):
    df = df.copy()

    # Drop colonnes trop vides
    cols_to_drop = ['ZONE1', 'ZONE2']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Variables activité → 0
    activity_cols = ['DATA_VOLUME', 'FREQUENCE_RECH', 'MONTANT']
    for col in activity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Numériques → médiane + flag missing
    num_cols = ['REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'TENURE']
    for col in num_cols:
        if col in df.columns:
            df[col + "_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())

    # Catégorielles
    if 'REGION' in df.columns:
        df['REGION'] = df['REGION'].fillna('Unknown')

    # Drop ID
    if 'user_id' in df.columns:
        df.drop(columns=['user_id'], inplace=True)

    return df


def encode_features(df):
    df = df.copy()

    if 'REGION' in df.columns:
        df = pd.get_dummies(df, columns=['REGION'], drop_first=True)

    return df


def prepare_data(df):
    df = clean_data(df)
    df = encode_features(df)
    return df