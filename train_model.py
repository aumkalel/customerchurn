"""
Train a Gradient Boosting model for customer churn prediction.
Saves model_new.pkl and scaler_new.pkl for use by the Flask app.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# === Encoding Maps (must match app.py) ===
ENCODING_MAP = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'OnlineBackup': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'DeviceProtection': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'TechSupport': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingTV': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'StreamingMovies': {'No internet service': 0, 'No': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    }
}

FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def main():
    # Load data
    df = pd.read_csv('customer churn.csv')
    print(f"Loaded {len(df)} records")

    # Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode all categorical columns
    for col, mapping in ENCODING_MAP.items():
        df[col] = df[col].map(mapping)

    # Features and target
    X = df[FEATURE_COLUMNS].copy()
    y = df['Churn']

    # Scale numerical features
    scaler = StandardScaler()
    X[NUMERICAL_COLUMNS] = scaler.fit_transform(X[NUMERICAL_COLUMNS])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    print(f"\nTrain accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test), target_names=['No Churn', 'Churn']))

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
    print("\nTop Feature Importances:")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat}: {imp:.4f}")

    # Save
    with open('model_new.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_new.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n✅ Model saved as model_new.pkl")
    print("✅ Scaler saved as scaler_new.pkl")


if __name__ == '__main__':
    main()
