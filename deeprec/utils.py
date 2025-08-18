import os, random
import torch
import numpy as np
import polars as pl
from tmp.feature import feature_engineering, feature_selection


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_cat_dims(X, cat_features):
    cat_dims = {}
    for c in cat_features:
        max_val = X[c].max()
        if max_val < 0:
            max_val = 0
        cat_dims[c] = max_val + 1
    return cat_dims


def normalize_numeric_features(
    X,
    numeric_features,
    train_mask,
    method="standard",
    clip_minmax_quantiles=(0.05, 0.95),
):
    for col in numeric_features:
        values = X[col].to_numpy()
        train_vals = values[train_mask]

        # 如果需要剪枝，先计算分位数并裁剪
        if clip_minmax_quantiles is not None:
            low_q, high_q = clip_minmax_quantiles
            low_val = np.quantile(train_vals, low_q)
            high_val = np.quantile(train_vals, high_q)
            # 对全部values做剪枝
            values = np.clip(values, low_val, high_val)
            train_vals = np.clip(train_vals, low_val, high_val)

        if method == "standard":
            mean = train_vals.mean()
            std_val = train_vals.std()
            std = std_val if std_val > 0 else 1.0
            norm_vals = (values - mean) / std

        elif method == "minmax":
            min_val = train_vals.min()
            max_val = train_vals.max()
            denom = max_val - min_val if max_val > min_val else 1.0
            norm_vals = (values - min_val) / denom

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        X = X.with_columns(pl.Series(col, norm_vals).alias(col))
    return X


def load_or_prepare_data(data_dir, full):
    cache_file = "feateng_not_full.parquet" if not full else "feateng_full.parquet"
    cache_path = os.path.join(
        "/home/zhengxiang/FlightRank/20250727_084025/data", cache_file
    )

    train_file = "train.parquet"
    test_file = "test.parquet"

    if os.path.exists(cache_path):
        print(f"Loading cached feature-engineered data from {cache_path} ...")
        df = pl.read_parquet(cache_path)

        test = pl.read_parquet(os.path.join(data_dir, test_file)).drop(
            "__index_level_0__"
        )
        test_ids = test["Id"]
        test_rankers = test["ranker_id"]
    else:
        print("Cached file not found, processing raw data ...")

        train = pl.read_parquet(os.path.join(data_dir, train_file)).drop(
            "__index_level_0__"
        )
        test = (
            pl.read_parquet(os.path.join(data_dir, test_file))
            .drop("__index_level_0__")
            .with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))
        )

        test_ids = test["Id"]
        test_rankers = test["ranker_id"]

        df = pl.concat((train, test))
        df = feature_engineering(df, full=True)

        df = df.with_columns(
            [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns]
            + [
                pl.col(c).fill_null("missing")
                for c in df.select(pl.selectors.string()).columns
            ]
        )

        os.makedirs(data_dir, exist_ok=True)
        df.write_parquet(cache_path)
        print(f"Saved feature-engineered data to {cache_path}")

    return df, test_ids, test_rankers


def prepare_data(data_dir, full=False):
    df, test_ids, test_rankers = load_or_prepare_data(data_dir, full)

    id_column = df["Id"]
    X, y, groups, cat_features, num_features = feature_selection(df)

    X = X.with_columns(pl.Series("Id", id_column[: X.shape[0]]))

    X = X.with_columns(
        [
            (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32)
            for c in cat_features
        ]
    )
    if not full:
        train_mask = np.arange(X.height) < 14510026
    else:
        train_mask = np.arange(X.height) < 18145372
    X = normalize_numeric_features(X, num_features, train_mask, "standard")

    return X, y, groups, cat_features, num_features, test_ids, test_rankers
