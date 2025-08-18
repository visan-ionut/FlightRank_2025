import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
from tqdm import tqdm
from transformers import get_scheduler

from deeprec.data import RankDataset, collate_fn
from deeprec.model import DLRankerAttention, DLRankerDeepTransformer, DLRanker
from deeprec.validation import hitrate_at_k, ndcg_at_k, map_at_k
from deeprec.utils import *
from deeprec.loss import xranknet_loss, softmax_ce_loss, ranking_hinge_loss


def train_one_epoch(model, loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (x_cat, x_num, y, lengths) in enumerate(pbar):
        x_num, y = x_num.to(device), y.to(device)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths)
        lengths = lengths.detach().clone().to(device)
        x_cat = {k: v.to(device) for k, v in x_cat.items()}

        optimizer.zero_grad()
        scores = model(x_cat, x_num)
        loss = xranknet_loss(scores, y, lengths)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        avg_loss = total_loss / (i + 1)
        pbar.set_postfix({"loss": loss.item(), "avg_loss": avg_loss})

    return total_loss / len(loader)


def validate(model, loader, device, min_group_size=10):
    model.eval()
    total_hitrate = 0.0
    total_ndcg = 0.0
    total_map = 0.0
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x_cat, x_num, y, lengths in tqdm(loader, desc="Validating", leave=False):
            x_num, y = x_num.to(device), y.to(device)
            if not torch.is_tensor(lengths):
                lengths = torch.tensor(lengths, device=device)
            else:
                lengths = lengths.to(device)
            x_cat = {k: v.to(device) for k, v in x_cat.items()}

            scores = model(x_cat, x_num)

            hitrate = hitrate_at_k(
                scores, y, lengths, k=3, min_group_size=min_group_size
            )
            ndcg = ndcg_at_k(scores, y, lengths, k=3, min_group_size=min_group_size)
            map3 = map_at_k(scores, y, lengths, k=3, min_group_size=min_group_size)
            loss = xranknet_loss(scores, y, lengths)

            batch_size = scores.size(0)
            total_hitrate += hitrate * batch_size
            total_ndcg += ndcg * batch_size
            total_map += map3 * batch_size
            total_loss += loss.item() * batch_size
            count += batch_size

    return (
        total_loss / count,
        total_hitrate / count,
        total_ndcg / count,
        total_map / count,
    )


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.counter = 0

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def main():
    DATA_DIR = "/home/zhengxiang/FlightRank/data"
    MODEL_DIR = "model"
    SUBMIT_DIR = "submission"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMIT_DIR, exist_ok=True)

    set_seed(42)

    best_val_loss = float("inf")
    best_hitrate = 0.0
    model_path = os.path.join(MODEL_DIR, "dl_transformer.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    val_hitrates = []
    val_ndcgs = []
    val_maps = []
    learning_rates = []

    early_stopper = EarlyStopping(patience=3, min_delta=0.001)

    # Load data and feature engineering
    X, y, groups, cat_features, num_features, test_ids, test_rankers = prepare_data(
        DATA_DIR
    )

    n1 = 14510026
    n2 = 18145372

    # Train-Val-Test split
    X_tr, X_va, X_te = X[:n1], X[n1:n2], X[n2:]
    y_tr, y_va, y_te = y[:n1], y[n1:n2], y[n2:]
    groups_tr, groups_va, groups_te = groups[:n1], groups[n1:n2], groups[n2:]

    cat_dims = build_cat_dims(X, cat_features)
    num_numeric_feats = len(num_features)

    train_dataset = RankDataset(X_tr, y_tr, groups_tr, cat_features, num_features)
    val_dataset = RankDataset(X_va, y_va, groups_va, cat_features, num_features)
    # test_dataset = RankDataset(X_te, y_te, groups_te, cat_features, num_features)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
    # )

    print("✅ Data loaders created.")

    # Configurations
    num_epochs = 10
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    learning_rate = 1e-3

    model = DLRanker(cat_dims, num_numeric_feats).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Model training
    best_hitrate = 0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_hitrate, val_ndcg, val_map = validate(
            model, val_loader, device, min_group_size=10
        )

        torch.cuda.empty_cache()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_hitrates.append(val_hitrate)
        val_ndcgs.append(val_ndcg)
        val_maps.append(val_map)

        current_lr = scheduler.get_last_lr()
        learning_rates.append(current_lr[0])

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"HitRate@3={val_hitrate:.4f}, "
            f"NDCG@3={val_ndcg:.4f}, "
            f"MAP@3={val_map:.4f}, "
            f"LR={current_lr[0]:.6f} "
            f"({elapsed:.1f}s)"
        )

        if val_loss < best_val_loss and val_hitrate > best_hitrate:
            best_val_loss = val_loss
            best_hitrate = val_hitrate
            torch.save(model.state_dict(), model_path)
            print("💾 Model improved (loss ↓ and hitrate ↑), saved.")

        # if early_stopper.step(-val_loss):
        #     print("⛔ Early stopping triggered.")
        #     break

    # Predict test set
    model.load_state_dict(torch.load(model_path))
    model.eval()

    val_ids = X_va["Id"]
    val_rankers = groups_va["ranker_id"]
    val_labels = y_va

    all_val_scores = []

    with torch.no_grad():
        for x_cat, x_num, _, lengths in tqdm(
            val_loader, desc="Predicting validation set"
        ):
            x_num = x_num.to(device)
            x_cat = {k: v.to(device) for k, v in x_cat.items()}
            scores = model(x_cat, x_num)

            for i in range(scores.size(0)):
                l = lengths[i]
                all_val_scores.append(scores[i, :l].cpu().numpy())

    all_val_scores = np.concatenate(all_val_scores)

    assert (
        len(all_val_scores) == X_va.shape[0]
    ), f"Mismatch: {len(all_val_scores)} scores vs {X_va.shape[0]} rows"

    val_df = pl.DataFrame(
        {
            "Id": val_ids,
            "ranker_id": val_rankers,
            "score": all_val_scores,
            "label": val_labels,
        }
    )

    val_df = val_df.with_columns(
        pl.col("score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("selected")
    ).select(["Id", "ranker_id", "score", "selected", "label"])

    val_save_path = os.path.join(MODEL_DIR, "validation_preds_dl.csv")
    val_df.write_csv(val_save_path)
    print(f"✅ Saved validation predictions with labels to {val_save_path}")


if __name__ == "__main__":
    main()
