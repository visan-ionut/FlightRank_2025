import time
import polars as pl
from functools import wraps
from datetime import datetime
from typing import Dict


TRAIN_VAL_SIZE = 16487352
# TRAIN_VAL_SIZE = 14510026
TRAIN_FLL_SIZE = 18145372


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result

    return wrapper


def hitrate_at_3(submission: pl.DataFrame, y_true, groups):
    df = pl.DataFrame(
        {"group": groups, "selected": submission["selected"], "true": y_true}
    )

    return (
        df.filter(pl.col("group").count().over("group") > 10)
        .sort(["group", "selected"], descending=[False, False])
        .group_by("group", maintain_order=True)
        .head(3)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )


def evaluate_hitrate_at_3(val, dval, y_va, groups_va, model, rerank=False):
    submission = make_submission(val, dval, model, "", score=False, rerank=rerank)
    xgb_hr3 = hitrate_at_3(submission, y_va, groups_va)
    print(f"HitRate@3: {xgb_hr3:.5f}")
    return model.predict(dval)


def make_submission(test, dtest, model, path, score=False, rerank=False):
    selected_list = (
        ["Id", "ranker_id", "selected"]
        if not score and not rerank
        else ["Id", "ranker_id", "selected", "pred_score"]
    )
    submission_xgb: pl.DataFrame = (
        test.select(["Id", "ranker_id"])
        .with_columns(pl.Series("pred_score", model.predict(dtest)))
        .with_columns(
            pl.col("pred_score")
            .rank(method="ordinal", descending=True)
            .over("ranker_id")
            .cast(pl.Int32)
            .alias("selected")
        )
        .select(selected_list)
    )

    if rerank:
        print(submission_xgb.head())
        top5 = re_rank(test, submission_xgb)

        submission_xgb = (
            submission_xgb.join(top5, on=["Id", "ranker_id"], how="left")
            .with_columns(
                [
                    pl.when(pl.col("new_selected").is_not_null())
                    .then(pl.col("new_selected"))
                    .otherwise(pl.col("selected"))
                    .alias("selected")
                ]
            )
            .select(["Id", "ranker_id", "selected"] + (["pred_score"] if score else []))
        )

    print(submission_xgb.head())
    if path != "":
        submission_xgb.write_parquet(path)
        print(f"Saved submission to: {path}")
    return submission_xgb


def hitrate_at_3_verbose(y_true, y_pred, groups, features_df=None):
    """
    Calculate HitRate@3 and print the groups that failed (no true == 1 in top 3).

    Parameters:
    - y_true: list or array of ground-truth labels (0/1)
    - y_pred: list or array of predicted scores
    - groups: list or array of group ids
    - features_df: optional, a polars DataFrame that includes the same row order as y_true/y_pred, and includes all relevant features.
    """
    df = pl.DataFrame({"group": groups, "pred": y_pred, "true": y_true})
    df = df.with_columns(pl.Series("row_idx", range(len(df))))

    # Filter only groups with size > 10
    df = df.filter(pl.col("group").count().over("group") > 10)

    # Get top 3 candidates per group
    top3 = (
        df.sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(3)
    )

    # Identify hit or miss per group
    hit_stats = top3.group_by("group").agg(pl.col("true").max().alias("hit"))

    # Join back to get failed groups
    failed_groups = hit_stats.filter(pl.col("hit") == 0).select("group")

    if failed_groups.height > 0:
        print(f"{failed_groups.height} groups missed Hit@3")

        failed_full = df.join(failed_groups, on="group", how="inner")

        if features_df is not None:
            # Add features (assumes same order)
            features_df = features_df.with_columns(
                pl.Series("row_idx", range(len(features_df)))
            )
            failed_full = failed_full.join(features_df, on="row_idx", how="left")
            failed_full = failed_full.drop("row_idx")

            print(failed_full)
        else:
            print(failed_full)
    else:
        print("All groups hit Hit@3.")

    # Calculate and return Hit@3
    return hit_stats["hit"].mean(), failed_full


def re_rank(test: pl.DataFrame, submission_xgb: pl.DataFrame, penalty_factor=0.1):
    COLS_TO_COMPARE = [
        "legs0_departureAt",
        "legs0_arrivalAt",
        "legs1_departureAt",
        "legs1_arrivalAt",
        "legs0_segments0_flightNumber",
        "legs1_segments0_flightNumber",
        # "legs0_segments0_aircraft_code",
        # "legs1_segments0_aircraft_code",
        # "legs0_segments0_departureFrom_airport_iata",
        # "legs1_segments0_departureFrom_airport_iata",
    ]

    test = test.with_columns(
        [pl.col(c).cast(str).fill_null("NULL") for c in COLS_TO_COMPARE]
    )

    df = submission_xgb.join(test, on=["Id", "ranker_id"], how="left")

    df = df.with_columns(
        (
            pl.col("legs0_departureAt")
            + "_"
            + pl.col("legs0_arrivalAt")
            + "_"
            + pl.col("legs1_departureAt")
            + "_"
            + pl.col("legs1_arrivalAt")
            + "_"
            + pl.col("legs0_segments0_flightNumber")
            + "_"
            + pl.col("legs1_segments0_flightNumber")
        ).alias("flight_hash")
    )

    # 计算每个航班 hash 的最大得分
    df = df.with_columns(
        pl.max("pred_score")
        .over(["ranker_id", "flight_hash"])
        .alias("max_score_same_flight")
    )

    # 计算惩罚后的分数（平滑）
    df = df.with_columns(
        (
            pl.col("pred_score")
            - penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score"))
        ).alias("reorder_score")
    )

    df = df.with_columns(
        pl.col("reorder_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("new_selected")
    )

    return df.select(["Id", "ranker_id", "new_selected", "pred_score", "reorder_score"])


SIMILARITY_COLS = {
    "legs0_departureAt": "time",
    "legs0_arrivalAt": "time",
    "legs1_departureAt": "time",
    "legs1_arrivalAt": "time",
    "legs0_segments0_flightNumber": "cat",
    "legs1_segments0_flightNumber": "cat",
    "legs0_segments0_aircraft_code": "cat",
    "legs1_segments0_aircraft_code": "cat",
}

TIME_SIM_THRESHOLD_HOURS = 24
LAMBDA = 0.7


def parse_time(t_str: str) -> float:
    # 转换为时间戳秒数，异常返回0
    try:
        dt = datetime.fromisoformat(t_str)
        return dt.timestamp()
    except Exception:
        return 0.0


def similarity(cand: Dict, sel: Dict) -> float:
    sim_sum = 0.0
    weight_sum = 0.0
    for col, typ in SIMILARITY_COLS.items():
        weight = 1.0  # 可以为不同字段定义不同权重
        weight_sum += weight
        if typ == "time":
            t1 = parse_time(cand.get(col, ""))
            t2 = parse_time(sel.get(col, ""))
            diff_hours = abs(t1 - t2) / 3600
            sim = max(0, 1 - diff_hours / TIME_SIM_THRESHOLD_HOURS)
            sim_sum += sim * weight
        else:
            sim = 1.0 if cand.get(col, None) == sel.get(col, None) else 0.0
            sim_sum += sim * weight
    return sim_sum / weight_sum if weight_sum > 0 else 0.0


def mmr_rerank(
    test: pl.DataFrame, submission_xgb: pl.DataFrame, top_k=10, lambda_=LAMBDA
) -> pl.DataFrame:
    for col in SIMILARITY_COLS.keys():
        if col in test.columns:
            test = test.with_columns(pl.col(col).cast(str).fill_null("NULL"))

    df = submission_xgb.join(test, on=["Id", "ranker_id"], how="left")

    result = []
    # groupby ranker_id
    for ranker_id, group_df in df.group_by("ranker_id"):
        candidates = group_df.to_dicts()
        selected = []

        while candidates and len(selected) < top_k:
            if not selected:
                # 第一个直接选 pred_score 最高
                best = max(candidates, key=lambda x: x["pred_score"])
                selected.append(best)
                candidates.remove(best)
            else:

                def mmr_score(cand):
                    sim_max = max(similarity(cand, sel) for sel in selected)
                    return lambda_ * cand["pred_score"] - (1 - lambda_) * sim_max

                best = max(candidates, key=mmr_score)
                selected.append(best)
                candidates.remove(best)

        # 给选中的分配 new_selected 排名
        for rank, item in enumerate(selected, 1):
            item["new_selected"] = rank
        result.extend(selected)

    return pl.DataFrame(result).select(
        ["Id", "ranker_id", "new_selected", "pred_score"]
    )
