from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)

    return model, score


def save_model(model, path="models/model.pkl"):
    joblib.dump(model, path)


def load_model(path="models/model.pkl"):
    return joblib.load(path)