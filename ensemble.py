import polars as pl
import glob
import re
import numpy as np
import pandas as pd


def load_submission_with_confidence(filepath):
    df = pl.read_csv(filepath)
    if "selected" in df.columns:
        df = df.drop("selected")
    return df


def compute_entropy(probabilities):
    p = np.clip(probabilities, 1e-9, 1.0)
    return -np.sum(p * np.log(p))


def pca_svd(X, n_components):
    """
    X: (n_samples, n_models) already standardized by columns.
    Returns:
      scores  : (n_samples, n_components)  = X @ Vt_k.T
      loadings: (n_models,  n_components)  = Vt_k.T
      recon   : (n_samples, n_models) reconstructed from the first components
    """
    # SVD on standardized X (centering + scaling should be done beforehand)
    # Note: np.linalg.svd returns Vt with shape (n_models, n_models)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Vt_k = Vt[:n_components, :]  # (k, n_models)
    scores = X @ Vt_k.T  # (n_samples, k)
    recon = scores @ Vt_k  # (n_samples, n_models)
    loadings = Vt_k.T  # (n_models, k)
    return scores, loadings, recon


def main():
    # Find all files matching *_with_confidence.csv
    filepaths = glob.glob("submission_*_*_with_confidence.csv")

    dfs = []
    confidence_col_names = []

    for filepath in filepaths:
        match = re.search(r"submission_.*_([\d.]+)_with_confidence\.csv", filepath)
        if match:
            score = match.group(1)
            df = load_submission_with_confidence(filepath)
            confidence_col = f"confidence_{score}"
            df = df.with_columns(pl.col("confidence").alias(confidence_col)).drop(
                "confidence"
            )
            confidence_col_names.append(confidence_col)
            dfs.append(df)

    if not dfs:
        print("❌ No valid submission files found.")
        return

    # Join on Id + ranker_id
    df_combined = dfs[0]
    for df in dfs[1:]:
        df_combined = df_combined.join(df, on=["Id", "ranker_id"])

    # Confidence matrix (n_samples x n_models)
    X_df = df_combined.select(confidence_col_names).to_pandas()
    # Replace NaN with column mean
    X_df = X_df.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X_df.values.astype(float)

    # Standardize columns (center and scale) for PCA
    col_means = X.mean(axis=0, keepdims=True)
    col_stds = X.std(axis=0, ddof=1, keepdims=True)
    col_stds[col_stds == 0.0] = 1.0  # protection against zero std
    X_std = (X - col_means) / col_stds

    # PCA parameters + Spearman calculation mode
    k_pca = 5
    spearman_mode = "residual"  # "residual" (recommended) | "loadings" | "scores"

    # PCA via SVD
    scores, loadings, recon = pca_svd(X_std, n_components=k_pca)

    if spearman_mode == "residual":
        # Idiosyncratic signal (after removing the first k components)
        X_resid = X_std - recon
        corr_df = pd.DataFrame(X_resid, columns=confidence_col_names).corr(
            method="spearman"
        )
        print(f"📊 Spearman between models on RESIDUALS after PCA (k={k_pca}):")
    elif spearman_mode == "loadings":
        # Spearman correlation between each model's loadings on the first k components
        load_df = pd.DataFrame(
            loadings,
            index=confidence_col_names,
            columns=[f"PC{i+1}" for i in range(k_pca)],
        )
        corr_df = load_df.T.corr(method="spearman")
        print(f"📊 Spearman between models on PCA LOADINGS (first {k_pca}):")
    elif spearman_mode == "scores":
        # Spearman correlation between models viewed through the PCA scores
        # (correlate reconstructed model columns from the first k components)
        X_k = recon  # common signal from the first components
        corr_df = pd.DataFrame(X_k, columns=confidence_col_names).corr(
            method="spearman"
        )
        print(f"📊 Spearman between models on PCA SCORES (first {k_pca}):")
    else:
        raise ValueError(
            "Invalid spearman_mode. Choose: 'residual' | 'loadings' | 'scores'."
        )

    print(corr_df)

    # Uniqueness based on 1 - average correlation (exclude diagonal = 1)
    mean_corr = corr_df.apply(lambda row: (row.sum() - 1) / (len(row) - 1), axis=1)
    model_uniqueness = 1 - mean_corr
    uniqueness_weights = model_uniqueness / model_uniqueness.sum()

    # Entropy for each confidence column (on raw scores, not standardized)
    entropies = []
    for col in confidence_col_names:
        probs = df_combined[col].to_numpy()
        probs = np.nan_to_num(probs, nan=np.nanmean(probs))
        total = np.sum(probs)
        if total <= 0:
            # stable fallback
            probs = np.full_like(probs, 1.0 / len(probs), dtype=float)
        else:
            probs = probs / total
        entropies.append(compute_entropy(probs))

    entropy_weights = np.array(entropies, dtype=float)
    entropy_weights /= entropy_weights.sum()

    # Combine weights: uniqueness (post-PCA) + entropy (informativeness)
    alpha = 0.75  # more emphasis on uniqueness
    combined_weights = alpha * uniqueness_weights.values + (1 - alpha) * entropy_weights
    combined_weights = combined_weights / combined_weights.sum()

    print("\n⚖️ Final ensemble weights (uniqueness + entropy):")
    for col, weight in zip(confidence_col_names, combined_weights):
        print(f"{col}: {weight:.4f}")

    # Weighted ensemble on confidence
    weighted_conf = sum(
        df_combined[col].fill_null(0) * weight
        for col, weight in zip(confidence_col_names, combined_weights)
    )
    df_combined = df_combined.with_columns(weighted_conf.alias("ensemble_confidence"))

    # Ranking per group (ranker_id) by ensemble_confidence
    df_ranked = df_combined.with_columns(
        pl.col("ensemble_confidence")
        .rank("ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("selected")
    )

    final_submission = df_ranked.select(["Id", "ranker_id", "selected"])

    print("\n✅ Sample of final submission:")
    print(final_submission.head())

    final_submission.write_parquet("submission_ensemble.parquet")
    print("\n💾 Saved as: submission_ensemble.parquet")


if __name__ == "__main__":
    main()
