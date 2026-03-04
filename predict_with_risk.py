import pandas as pd
import joblib
import numpy as np
import os

# =========================
# Paths
# =========================
MODEL_PATH = "models/lgbm_model.joblib"
X_PATH = "data/processed/X_features_enriched.csv"
META_PATH = "data/processed/X_clean.csv"   # contains chromosome, gene, position
OUT_PATH = "predictions_with_risk.csv"

# =========================
# Thresholds (from PR tuning)
# =========================
DECISION_THRESHOLD = 0.54   # tuned threshold

LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.70


# =========================
# Risk assignment
# =========================
def assign_risk(prob):
    if prob < LOW_RISK_MAX:
        return "Low risk"
    elif prob >= HIGH_RISK_MIN:
        return "High risk"
    else:
        return "Uncertain"


def predict_with_risk():
    print("🔹 Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("🔹 Loading feature matrix...")
    X = pd.read_csv(X_PATH, low_memory=False)

    print("🔹 Loading variant metadata...")
    meta = pd.read_csv(META_PATH, low_memory=False)

    # =========================
    # Align lengths safely
    # =========================
    min_len = min(len(X), len(meta))
    X = X.iloc[:min_len].reset_index(drop=True)
    meta = meta.iloc[:min_len].reset_index(drop=True)

    print(f"🔹 Aligned to {min_len} variants")

    # =========================
    # Predict probabilities
    # =========================
    print("🔹 Predicting pathogenic probabilities...")
    probs = model.predict_proba(X)[:, 1]

    binary_pred = (probs >= DECISION_THRESHOLD).astype(int)

    # =========================
    # Build final output
    # =========================
    results = pd.DataFrame({
        "chromosome": meta["chromosome"],
        "position": meta["position"],
        "gene": meta["gene"],
        "pathogenic_probability": probs,
        "binary_prediction": binary_pred,
        "risk_category": [assign_risk(p) for p in probs]
    })

    results.to_csv(OUT_PATH, index=False)

    print("✅ Predictions saved to:", OUT_PATH)
    print("\n📊 Risk category distribution:")
    print(results["risk_category"].value_counts())


if __name__ == "__main__":
    predict_with_risk()
