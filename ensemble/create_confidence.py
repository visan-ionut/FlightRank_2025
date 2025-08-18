import polars as pl


def compute_confidence(submission_path):
    # Read the submission CSV file
    df = pl.read_parquet(submission_path)

    # Use pl.len() instead of deprecated pl.count()
    df = df.with_columns([pl.len().over("ranker_id").alias("group_size")])

    # Compute normalized confidence from selected position and group size
    df = df.with_columns(
        [
            (
                1.0 - ((pl.col("selected") - 1) / (pl.col("group_size") - 1 + 1e-8))
            ).alias("confidence")
        ]
    )

    # Optionally drop intermediate column
    return df.drop("group_size")


if __name__ == "__main__":

    time_tag = "20250815103211"

    df_1 = compute_confidence(f"./submission/submission_{time_tag}.parquet")
    print(df_1.head())
    df_1.write_parquet(f"./ensemble/submission_{time_tag}_with_confidence.parquet")
