import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_data, preprocess_data, split_data

def train_model():
    # Load and preprocess data
    df = load_data("data/creditcard.csv")
    X_scaled, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Model and scaler saved successfully!")

# If you want to auto-run when executing the script directly:
if __name__ == "__main__":
    train_model()