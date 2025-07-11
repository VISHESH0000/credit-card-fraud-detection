import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, preprocess_data, split_data

def train_model():
    # Load and preprocess datapython src/train.py
    df = load_data("data/creditcard.csv")
    X_scaled, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("models/model.joblib", "wb") as f:
        joblib.dump(model, f, compress=3)

    # Save scaler
    with open("models/scaler.joblib", "wb") as f:
        joblib.dump(scaler, f, compress=3)

    print("✅ Model and scaler saved successfully ")

if __name__ == "__main__":
    train_model()