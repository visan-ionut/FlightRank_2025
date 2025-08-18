import os
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb

from .utils import TRAIN_VAL_SIZE


def split_dataset(
    train: pl.DataFrame,
    X: pl.DataFrame,
    y: pl.DataFrame,
    groups: pl.DataFrame,
    cat_features_final,
    model,
):
    X = X.with_columns(
        [
            (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16)
            for c in cat_features_final
        ]
    )

    # Create train/validation/test dataset
    n1 = TRAIN_VAL_SIZE
    n2 = train.height
    X_tr, X_va, X_te, X_fl = X[:n1], X[n1:n2], X[n2:], X[:n2]
    y_tr, y_va, y_te, y_fl = y[:n1], y[n1:n2], y[n2:], y[:n2]
    groups_tr, groups_va, groups_te, groups_fl = (
        groups[:n1],
        groups[n1:n2],
        groups[n2:],
        groups[:n2],
    )

    if model == "xgboost":
        group_sizes_tr = (
            groups_tr.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_va = (
            groups_va.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_te = (
            groups_te.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_fl = (
            groups_fl.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )

        dtrain = xgb.DMatrix(
            X_tr,
            label=y_tr,
            group=group_sizes_tr,
            # weight=weight_tr,
            feature_names=X.columns,
        )
        dval = xgb.DMatrix(
            X_va,
            label=y_va,
            group=group_sizes_va,
            # weight=weight_va,
            feature_names=X.columns,
        )
        dtest = xgb.DMatrix(
            X_te, label=y_te, group=group_sizes_te, feature_names=X.columns
        )
        dfull = xgb.DMatrix(
            X_fl,
            label=y_fl,
            group=group_sizes_fl,
            # weight=weight_fl,
            feature_names=X.columns,
        )
    elif model == "lightgbm":

        def get_group_sizes(groups):
            return (
                groups.group_by("ranker_id", maintain_order=True)
                .agg(pl.len())["len"]
                .to_list()
            )

        group_sizes_tr = get_group_sizes(groups_tr)
        group_sizes_va = get_group_sizes(groups_va)
        group_sizes_te = get_group_sizes(groups_te)
        group_sizes_fl = get_group_sizes(groups_fl)

        # Convert to pandas dataframe for compability
        X_tr = X_tr.to_pandas()
        X_va = X_va.to_pandas()
        X_te = X_te.to_pandas()
        X_fl = X_fl.to_pandas()

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr.to_numpy().flatten(),
            group=group_sizes_tr,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va,
            label=y_va.to_numpy().flatten(),
            group=group_sizes_va,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dtest = lgb.Dataset(
            X_te,
            label=y_te.to_numpy().flatten(),
            group=group_sizes_te,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dfull = lgb.Dataset(
            X_fl,
            label=y_fl.to_numpy().flatten(),
            group=group_sizes_fl,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
    else:
        raise ValueError("Unsupported Models!")

    return (
        dtrain,
        dval,
        dtest,
        dfull,
        X_tr,
        y_tr,
        groups_tr,
        X_va,
        y_va,
        groups_va,
        X_te,
    )


def split_dataset_leg(
    train: pl.DataFrame,
    X: pl.DataFrame,
    y: pl.DataFrame,
    groups: pl.DataFrame,
    cat_features_final: list,
    legs: int,
):
    """
    legs: 1 表示单 leg, 2 表示双 leg
    is_one_way = 1 代表单 leg, 0 代表双 leg（需确认你的数据定义）
    """

    # 编码分类特征
    X = X.with_columns(
        [
            (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16)
            for c in cat_features_final
        ]
    )

    # ---- 先按照原顺序划分 ----
    n1 = TRAIN_VAL_SIZE
    n2 = train.height
    X_tr, X_va, X_te, X_fl = X[:n1], X[n1:n2], X[n2:], X[:n2]
    y_tr, y_va, y_te, y_fl = y[:n1], y[n1:n2], y[n2:], y[:n2]
    groups_tr, groups_va, groups_te, groups_fl = (
        groups[:n1],
        groups[n1:n2],
        groups[n2:],
        groups[:n2],
    )

    # ---- 再用 mask 过滤单/双 leg ----
    if legs == 1:
        mask_tr = X_tr["is_one_way"] == 1
        mask_va = X_va["is_one_way"] == 1
        mask_te = X_te["is_one_way"] == 1
        mask_fl = X_fl["is_one_way"] == 1
    elif legs == 2:
        mask_tr = X_tr["is_one_way"] == 0
        mask_va = X_va["is_one_way"] == 0
        mask_te = X_te["is_one_way"] == 0
        mask_fl = X_fl["is_one_way"] == 0
    else:
        raise ValueError("legs must be 1 or 2")

    X_tr, y_tr, groups_tr = (
        X_tr.filter(mask_tr).drop("is_one_way"),
        y_tr.filter(mask_tr),
        groups_tr.filter(mask_tr),
    )
    X_va, y_va, groups_va = (
        X_va.filter(mask_va).drop("is_one_way"),
        y_va.filter(mask_va),
        groups_va.filter(mask_va),
    )
    X_te, y_te, groups_te = (
        X_te.filter(mask_te).drop("is_one_way"),
        y_te.filter(mask_te),
        groups_te.filter(mask_te),
    )
    X_fl, y_fl, groups_fl = (
        X_fl.filter(mask_fl).drop("is_one_way"),
        y_fl.filter(mask_fl),
        groups_fl.filter(mask_fl),
    )

    # ---- XGBoost 排序任务的 group size ----
    group_sizes_tr = (
        groups_tr.group_by("ranker_id", maintain_order=True)
        .agg(pl.len())["len"]
        .to_numpy()
    )
    group_sizes_va = (
        groups_va.group_by("ranker_id", maintain_order=True)
        .agg(pl.len())["len"]
        .to_numpy()
    )
    group_sizes_te = (
        groups_te.group_by("ranker_id", maintain_order=True)
        .agg(pl.len())["len"]
        .to_numpy()
    )
    group_sizes_fl = (
        groups_fl.group_by("ranker_id", maintain_order=True)
        .agg(pl.len())["len"]
        .to_numpy()
    )

    # ---- 构造 DMatrix ----
    dtrain = xgb.DMatrix(
        X_tr,
        label=y_tr,
        group=group_sizes_tr,
        feature_names=X_tr.columns,
    )
    dval = xgb.DMatrix(
        X_va,
        label=y_va,
        group=group_sizes_va,
        feature_names=X_va.columns,
    )
    dtest = xgb.DMatrix(
        X_te, label=y_te, group=group_sizes_te, feature_names=X_te.columns
    )
    dfull = xgb.DMatrix(
        X_fl,
        label=y_fl,
        group=group_sizes_fl,
        feature_names=X_fl.columns,
    )

    return (
        dtrain,
        dval,
        dtest,
        dfull,
        X_tr,
        y_tr,
        groups_tr,
        X_va,
        y_va,
        groups_va,
        X_te,
        mask_va,
        mask_te,
    )
