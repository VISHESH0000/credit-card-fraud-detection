import pickle
import pandas as pd
from sklearn.metrics import classification_report
from src.preprocess import load_data, preprocess_data, split_data

def evaluate_model():
    # Load dataset
    df = load_data("data/creditcard.csv")
    X_scaled, y, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Load trained model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("📊 Evaluation Report:")
    print(classification_report(y_test, y_pred))