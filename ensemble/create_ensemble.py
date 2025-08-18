import polars as pl


def load_submission_with_confidence(tag, alpha=0.7, k=5):
    print(tag)
    df = pl.read_parquet(f"./ensemble/submission_{tag}_with_confidence.parquet")

    rrf_max = 1 / (k + 1)
    df = df.with_columns(((1 / (k + pl.col("selected"))) / rrf_max).alias("rrf_score"))

    df = df.with_columns(
        (alpha * pl.col("confidence") + (1 - alpha) * pl.col("rrf_score")).alias(
            "confidence"
        )
    )

    df = df.drop(["selected", "rrf_score"])

    return df


if __name__ == "__main__":
    timetag_score = {
        # "20250716132706": [0, 0.51381],
        # "20250718083308": [0, 0.51822],
        # "20250719002505": [0, 0.51859],
        # "20250720003111": [0, "0.50693/lgb"],
        # "20250721025740": [0, 0.51960],
        # "20250722050939": [1, 0.52070],
        "20250721083807": [1, 0.52244],
        "20250724032338": [1.25, "0.51345/lgb"],
        # "20250725040223": [1, 0.52309],
        "20250725083055": [1, 0.52391],
        "20250727084025": [1, 0.52795],
        # "20250728094305": [1, 0.52492],
        # "20250731122023": [1, 0.52015],
        # "20250729084249": [1, 0.51822],
        "20250802074816": [1, 0.52603],
        "20250807032439": [1, 0.52538],
        "20250804001151": [1.25, 0.51244],  # lgb
        "dl_ranker": [2, 0.48755],
    }

    dfs = []
    for timetag, score_list in timetag_score.items():
        df = load_submission_with_confidence(timetag)
        weight, score = score_list
        print(f"{weight} * {score}")
        df = df.with_columns(
            (pl.col("confidence") * weight).alias("confidence_weighted")
        )
        dfs.append(df)

    # Combine all the dfs
    df_combined = (
        pl.concat(dfs)
        .group_by(["Id", "ranker_id"])
        .agg(pl.sum("confidence_weighted").alias("confidence_sum"))
    )

    # Rank within each ranker_id to generate 'selected'
    df_ranked = df_combined.with_columns(
        [
            pl.col("confidence_sum")
            .rank(method="ordinal", descending=True)
            .over("ranker_id")
            .cast(pl.Int32)
            .alias("selected")
        ]
    )

    # Load original order from one of the submissions
    df_original = dfs[0].select(["Id", "ranker_id"])

    # Join and keep only required columns in original order
    final_submission = df_original.join(
        df_ranked.select(["Id", "ranker_id", "selected"]),
        on=["Id", "ranker_id"],
        how="left",
    )
    print(final_submission.head())

    # Save to .parquet format
    final_submission.write_parquet("submission_ensemble.parquet")
    print(
        "✅ Ensemble saved as submission_ensemble.parquet with Id, ranker_id, selected only."
    )
