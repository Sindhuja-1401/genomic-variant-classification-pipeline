import pandas as pd

X_PATH = "data/processed/X_clean.csv"
Y_PATH = "data/processed/y_clean.csv"
OUT_PATH = "data/processed/gene_prior.csv"

def compute_gene_prior():
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)["label"]

    df = X.copy()
    df["label"] = y

    gene_prior = (
        df.groupby("gene")["label"]
        .mean()
        .reset_index()
        .rename(columns={"label": "gene_pathogenic_prior"})
    )

    gene_prior.to_csv(OUT_PATH, index=False)
    print("✅ Gene pathogenicity prior saved")

if __name__ == "__main__":
    compute_gene_prior()

