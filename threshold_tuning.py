import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    classification_report
)

# =========================
# Paths
# =========================
MODEL_PATH = "models/lgbm_model.joblib"
X_PATH = "data/processed/X_features_enriched.csv"
Y_PATH = "data/processed/y_clean.csv"

# =========================
# Config
# =========================
TARGET_RECALL = 0.95   # medical safety first


def tune_threshold():
    print("🔹 Loading model and data...")
    model = joblib.load(MODEL_PATH)

    X = pd.read_csv(X_PATH, low_memory=False)
    y = pd.read_csv(Y_PATH)["label"]

    # -------------------------
    # Align X and y (safety)
    # -------------------------
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)

    print(f"🔹 Aligned to {min_len} samples")

    # -------------------------
    # Predict probabilities
    # -------------------------
    probs = model.predict_proba(X)[:, 1]

    # -------------------------
    # Precision–Recall curve
    # -------------------------
    precision, recall, thresholds = precision_recall_curve(y, probs)

    # Plot PR curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (LightGBM)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # CORRECT threshold selection:
    # Recall ≥ TARGET_RECALL AND max precision
    # ---------------------------------------------------------
    valid_idx = np.where(recall[:-1] >= TARGET_RECALL)[0]

    if len(valid_idx) == 0:
        print("⚠️ No threshold meets target recall. Using default 0.5")
        chosen_threshold = 0.5
    else:
        best_idx = valid_idx[np.argmax(precision[valid_idx])]
        chosen_threshold = thresholds[best_idx]

    print(f"\n✅ Selected threshold (Recall ≥ {TARGET_RECALL} with max precision): "
          f"{chosen_threshold:.4f}")

    # -------------------------
    # Evaluate at tuned threshold
    # -------------------------
    preds = (probs >= chosen_threshold).astype(int)

    print("\n📊 Classification Report @ tuned threshold:")
    print(classification_report(y, preds, zero_division=0))

    roc_auc = roc_auc_score(y, probs)
    print(f"ROC-AUC (threshold-independent): {roc_auc:.4f}")

    return chosen_threshold


if __name__ == "__main__":
    tune_threshold()

