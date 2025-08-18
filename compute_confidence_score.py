import polars as pl
import os


def compute_confidence(
    df: pl.DataFrame, alpha: float = 0.7, k: int = 5
) -> pl.DataFrame:
    # group size for each ranker_id
    df = df.with_columns([pl.len().over("ranker_id").alias("group_size")])

    # 1) base score based on position within the group (max for selected = 1)
    base_conf = 1.0 - ((pl.col("selected") - 1) / (pl.col("group_size") - 1 + 1e-8))
    df = df.with_columns(base_conf.alias("confidence"))

    # 2) RRF score normalized to [0, 1]
    rrf_max = 1.0 / (k + 1)  # RRF value for selected = 1
    df = df.with_columns(
        ((1.0 / (k + pl.col("selected"))) / rrf_max).alias("rrf_score")
    )

    # 3) convex combination between base score and RRF
    df = df.with_columns(
        (alpha * pl.col("confidence") + (1 - alpha) * pl.col("rrf_score")).alias(
            "confidence"
        )
    )

    # keep 'selected'; remove only temporary columns
    return df.drop(["group_size", "rrf_score"])


def process_file(input_path: str, alpha: float = 0.7, k: int = 5):
    print(f"🔄 Processing: {input_path}")

    if input_path.endswith(".csv"):
        df = pl.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        df = pl.read_parquet(input_path)
    else:
        print(f"⚠️ Skipping unsupported file: {input_path}")
        return

    df_conf = compute_confidence(df, alpha=alpha, k=k)

    base, ext = os.path.splitext(input_path)
    if base.endswith("_with_confidence"):
        output_path = f"{base}.csv"
    else:
        output_path = f"{base}_with_confidence.csv"

    df_conf.write_csv(output_path)
    print(f"✅ Saved: {output_path}\n")


def main():
    files = [
        "submission_20250721083807_0.52244.parquet",
        "submission_20250724032338_0.51345.parquet",
        "submission_20250725083055_0.52391.parquet",
        "submission_20250727084025_0.52795.parquet",
        "submission_20250802074816_0.52603.parquet",
        "submission_20250804001151_0.51244.parquet",
        "submission_20250807032439_0.52538.parquet",
        "submission_dl_ranker_0.48755.parquet",
    ]

    alpha = 0.7
    k = 5

    for file in files:
        if os.path.exists(file):
            process_file(file, alpha=alpha, k=k)
        else:
            print(f"❌ File not found: {file}")


if __name__ == "__main__":
    main()
