import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# =========================
# Paths
# =========================
X_PATH = "data/processed/X_features_enriched.csv"
Y_PATH = "data/processed/y_clean.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.joblib")


def train_lightgbm():
    print("🔹 Loading data...")
    X = pd.read_csv(X_PATH, low_memory=False)
    y = pd.read_csv(Y_PATH)["label"]

    # =========================
    # CRITICAL FIX: ALIGN X & y
    # =========================
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)

    print(f"🔹 Aligned X and y to {min_len} samples")

    # =========================
    # Train / Validation Split
    # =========================
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🔹 Training LightGBM model...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    print("🔹 Evaluating model...")
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_val, preds))

    roc_auc = roc_auc_score(y_val, probs)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # =========================
    # Save model
    # =========================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_lightgbm()

