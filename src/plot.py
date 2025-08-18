import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def calculate_hitrate_curve(df, k_values):
    """Function to calculate hitrate curve efficiently"""
    # Sort once and calculate all k values
    sorted_df = df.sort(["ranker_id", "pred_score"], descending=[False, True])
    return [
        sorted_df.group_by("ranker_id", maintain_order=True)
        .head(k)
        .group_by("ranker_id")
        .agg(pl.col("selected").max().alias("hit"))
        .select(pl.col("hit").mean())
        .item()
        for k in k_values
    ]


def plot_hitrate_at_k(groups_va, xgb_va_preds, y_va):
    # Color palette
    red = (0.86, 0.08, 0.24)
    blue = (0.12, 0.56, 1.0)

    # Prepare data for analysis
    va_df = pl.DataFrame(
        {
            "ranker_id": groups_va.to_numpy().flatten(),
            "pred_score": xgb_va_preds,
            "selected": y_va.to_numpy().flatten(),
        }
    )

    # Add group size and filter
    va_df = va_df.join(
        va_df.group_by("ranker_id").agg(pl.len().alias("group_size")), on="ranker_id"
    ).filter(pl.col("group_size") > 10)

    # Calculate group size quantiles
    size_quantiles = (
        va_df.select("ranker_id", "group_size")
        .unique()
        .select(
            pl.col("group_size").quantile(0.25).alias("q25"),
            pl.col("group_size").quantile(0.50).alias("q50"),
            pl.col("group_size").quantile(0.75).alias("q75"),
        )
        .to_dicts()[0]
    )

    # Calculate curves
    k_values = list(range(1, 21))
    curves = {
        "All groups (>10)": calculate_hitrate_curve(va_df, k_values),
        f'Small (11-{int(size_quantiles["q25"])})': calculate_hitrate_curve(
            va_df.filter(pl.col("group_size") <= size_quantiles["q25"]), k_values
        ),
        f'Medium ({int(size_quantiles["q25"]+1)}-{int(size_quantiles["q75"])})': calculate_hitrate_curve(
            va_df.filter(
                (pl.col("group_size") > size_quantiles["q25"])
                & (pl.col("group_size") <= size_quantiles["q75"])
            ),
            k_values,
        ),
        f'Large (>{int(size_quantiles["q75"])})': calculate_hitrate_curve(
            va_df.filter(pl.col("group_size") > size_quantiles["q75"]), k_values
        ),
    }

    # Calculate hitrate@3 by group size using log-scale bins
    # Create log-scale bins
    min_size = va_df["group_size"].min()
    max_size = va_df["group_size"].max()
    bins = np.logspace(np.log10(min_size), np.log10(max_size), 51)  # 51 edges = 50 bins

    # Calculate hitrate@3 for each ranker_id
    ranker_hr3 = (
        va_df.sort(["ranker_id", "pred_score"], descending=[False, True])
        .group_by("ranker_id", maintain_order=True)
        .agg(
            [
                pl.col("selected").head(3).max().alias("hit_top3"),
                pl.col("group_size").first(),
            ]
        )
    )

    # Assign bins and calculate hitrate per bin
    bin_centers = (
        bins[:-1] + bins[1:]
    ) / 2  # Geometric mean would be more accurate for log scale
    bin_indices = np.digitize(ranker_hr3["group_size"].to_numpy(), bins) - 1

    size_analysis = (
        pl.DataFrame(
            {
                "bin_idx": bin_indices,
                "bin_center": bin_centers[
                    np.clip(bin_indices, 0, len(bin_centers) - 1)
                ],
                "hit_top3": ranker_hr3["hit_top3"],
            }
        )
        .group_by(["bin_idx", "bin_center"])
        .agg([pl.col("hit_top3").mean().alias("hitrate3"), pl.len().alias("n_groups")])
        .filter(pl.col("n_groups") >= 3)
        .sort("bin_center")
    )  # At least 3 groups per bin

    # Create combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=400)

    # Left plot: HitRate@k curves
    # Create color gradient from blue to red for size groups
    colors = ["black"]  # All groups is black
    for i in range(3):  # 3 size groups
        t = i / 2  # 0, 0.5, 1
        color = tuple(blue[j] * (1 - t) + red[j] * t for j in range(3))
        colors.append(color)

    for (label, hitrates), color in zip(curves.items(), colors):
        ax1.plot(k_values, hitrates, marker="o", label=label, color=color, markersize=3)
    ax1.set_xlabel("k (top-k predictions)")
    ax1.set_ylabel("HitRate@k")
    ax1.set_title("HitRate@k by Group Size")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)
    ax1.set_ylim(-0.025, 1.025)

    # Right plot: HitRate@3 vs Group Size (log scale)
    ax2.scatter(
        size_analysis["bin_center"],
        size_analysis["hitrate3"],
        s=30,
        alpha=0.6,
        color=blue,
    )
    ax2.set_xlabel("Group Size")
    ax2.set_ylabel("HitRate@3")
    ax2.set_title("HitRate@3 vs Group Size")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return curves


def plot_ndcg_curve(evals_result, metric):
    epochs = range(len(evals_result["train"][metric]))
    plt.plot(epochs, evals_result["train"][metric], label=f"Train {metric}")
    plt.plot(epochs, evals_result["val"][metric], label=f"Validation {metric}")
    plt.xlabel("Boosting Rounds")
    plt.ylabel(metric)
    plt.title(f"XGBoost {metric} over training rounds")
    plt.legend()
    plt.grid(True)
    plt.show()
