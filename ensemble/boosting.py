import polars as pl
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_conf_df(path: str) -> pl.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ File not found: {path}")
    df = pl.read_parquet(path).drop("selected")
    if "confidence" not in df.columns:
        raise ValueError(f"⚠️ Missing 'confidence' column in {path}")
    return df


def build_top_map(df: pl.DataFrame, top_k: int = 3) -> dict[str, dict[str, int]]:
    ranked = (
        df.with_columns(
            pl.col("confidence")
            .rank("ordinal", descending=True)
            .over("ranker_id")
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .select(["ranker_id", "Id", "rank"])
    )

    top_map = {}
    for row in ranked.iter_rows(named=True):
        rid, id_, rank = row["ranker_id"], row["Id"], int(row["rank"])
        if rid not in top_map:
            top_map[rid] = {}
        top_map[rid][id_] = rank
    return top_map


if __name__ == "__main__":
    V1 = "./ensemble/submission_20250718083308_with_confidence.parquet"
    V2 = "./ensemble/submission_20250721083807_with_confidence.parquet"
    V3 = "./ensemble/submission_dl_ranker_with_confidence.parquet"
    OUTPUT_FILE = "submission_ensemble_boosted.parquet"
    TOP_K = 5

    df_v1 = load_conf_df(V1)
    df_v2 = load_conf_df(V2)
    df_v3 = load_conf_df(V3)

    top_v1 = build_top_map(df_v1, TOP_K)
    top_v2 = build_top_map(df_v2, TOP_K)
    top_v3 = build_top_map(df_v3, TOP_K)

    df_combined = pl.concat([df_v1, df_v2, df_v3])
    df_grouped = df_combined.group_by(["Id", "ranker_id"]).agg(
        pl.sum("confidence").alias("conf_combined")
    )

    # Boost
    boosted_rows = []
    all_top = [build_top_map(df_v1), build_top_map(df_v2), build_top_map(df_v3)]

    for row in tqdm(df_grouped.iter_rows(named=True)):
        id_, ranker_id, conf = row["Id"], row["ranker_id"], row["conf_combined"]

        appear_count = 0
        rank_boosts = []
        for top_map in all_top:
            rank_dict = top_map.get(ranker_id, {})
            if id_ in rank_dict:
                appear_count += 1
                rank = rank_dict[id_]
                if rank == 1:
                    rank_boosts.append(1.25)
                elif rank == 2:
                    rank_boosts.append(1.2)
                elif rank == 3:
                    rank_boosts.append(1.15)
                else:
                    rank_boosts.append(1.1)

        # Base on rank and count at the same time
        count_boost = 1.5 if appear_count == 3 else (1.3 if appear_count == 2 else 1.0)
        rank_boost = sum(rank_boosts) / len(rank_boosts) if rank_boosts else 1.0
        boost = count_boost

        boosted_rows.append(
            {"Id": id_, "ranker_id": ranker_id, "confidence": conf * boost}
        )

    df_boosted = pl.DataFrame(boosted_rows)

    # ---------------------
    # RANK
    # ---------------------
    df_ranked = df_boosted.with_columns(
        [
            pl.col("confidence")
            .rank(method="ordinal", descending=True)
            .over("ranker_id")
            .cast(pl.Int32)
            .alias("selected")
        ]
    )

    # ---------------------
    # RESTORE ORIGINAL ORDER
    # ---------------------
    df_original = df_v1.select(["Id", "ranker_id"])
    final_submission = df_original.join(
        df_ranked.select(["Id", "ranker_id", "selected"]),
        on=["Id", "ranker_id"],
        how="left",
    )

    # ---------------------
    # SAVE
    # ---------------------
    final_submission.write_parquet(OUTPUT_FILE)
    print(final_submission.head())
    print(f"✅ Boosted ensemble saved to {OUTPUT_FILE}")
