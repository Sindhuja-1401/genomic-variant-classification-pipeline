import pandas as pd
import numpy as np

# Inputs
X_CLEAN = "data/processed/X_clean.csv"
GENE_PRIOR = "data/processed/gene_prior.csv"
CHROM_PRIOR = "data/processed/chrom_density.csv"

# Output
OUT = "data/processed/X_features_enriched.csv"


def encode_consequence(consequence):
    if pd.isna(consequence):
        return 0

    c = str(consequence).lower()

    if "frameshift" in c or "stop_gained" in c or "nonsense" in c:
        return 4
    if "splice" in c:
        return 3
    if "missense" in c:
        return 2
    if "synonymous" in c:
        return 1
    return 0

def encode_review(status):
    if pd.isna(status):
        return 0

    s = str(status).lower()
    if "practice_guideline" in s:
        return 4
    if "expert_panel" in s:
        return 3
    if "multiple_submitters" in s:
        return 2
    if "single_submitter" in s:
        return 1
    return 0


def engineer_enriched_features():
    print("🔹 Loading base cleaned data...")
    df = pd.read_csv(X_CLEAN, low_memory=False)

    print("🔹 Merging gene & chromosome priors...")
    gene_prior = pd.read_csv(GENE_PRIOR)
    chrom_prior = pd.read_csv(CHROM_PRIOR)

    df = df.merge(gene_prior, on="gene", how="left")
    df = df.merge(chrom_prior, on="chromosome", how="left")

    df.fillna(0, inplace=True)

    # --- Structural features ---
    df["ref_len"] = df["ref"].astype(str).apply(len)
    df["alt_len"] = df["alt"].astype(str).apply(len)
    df["indel_flag"] = (df["ref_len"] != df["alt_len"]).astype(int)

    # --- Frequency features ---
    df["af_exac"] = df["af_exac"].fillna(0)
    df["af_tgp"] = df["af_tgp"].fillna(0)
    df["af_esp"] = df["af_esp"].fillna(0)

    df["min_af"] = df[["af_exac", "af_tgp", "af_esp"]].min(axis=1)
    df["rare_variant_flag"] = (df["min_af"] < 0.01).astype(int)

    # --- Biological severity ---
    df["consequence_severity"] = df["molecular_consequence"].apply(encode_consequence)

    # --- Review strength ---
    df["review_strength"] = df["review_status"].apply(encode_review)

    # --- Chromosome encoding ---
    df["chromosome"] = df["chromosome"].astype(str)
    chrom_map = {str(i): i for i in range(1, 23)}
    chrom_map.update({"X": 23, "Y": 24, "MT": 25})
    df["chromosome_encoded"] = df["chromosome"].map(chrom_map).fillna(0)

    # --- Drop non-ML columns ---
    drop_cols = [
        "ref",
        "alt",
        "gene",
        "molecular_consequence",
        "review_status",
        "chromosome"
    ]
    df.drop(columns=drop_cols, inplace=True)

    df.to_csv(OUT, index=False)
    print("✅ Enriched feature engineering completed")
    print("📁 Saved to", OUT)
    print("🔹 Final shape:", df.shape)


if __name__ == "__main__":
    engineer_enriched_features()
