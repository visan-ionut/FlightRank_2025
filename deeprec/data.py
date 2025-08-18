import torch
import numpy as np
import polars as pl
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class MyRankDataset(Dataset):
    def __init__(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        groups: pl.DataFrame,
        cat_features: list[str],
        num_features: list[str],
        max_len: int = 2048,
    ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.X_cat = {c: X[c].to_numpy() for c in cat_features}
        self.X_num = X[self.num_features].to_numpy()
        self.y = y.to_numpy()
        self.groups = groups["ranker_id"].to_numpy()
        self.max_len = max_len

        self.group_to_indices = defaultdict(list)
        for i, g in enumerate(self.groups):
            self.group_to_indices[g].append(i)
        self.group_keys = list(self.group_to_indices.keys())

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        g = self.group_keys[idx]
        inds = self.group_to_indices[g]
        length = len(inds)

        # Select pos, sample neg
        if self.max_len is not None and length > self.max_len:
            pos_inds = [i for i in inds if self.y[i] > 0]
            neg_inds = [i for i in inds if self.y[i] == 0]

            remaining_slots = self.max_len - len(pos_inds)
            if remaining_slots > 0:
                sampled_neg = np.random.choice(neg_inds, remaining_slots, replace=False)
                inds = np.concatenate([pos_inds, sampled_neg])
            else:
                inds = np.array(pos_inds[: self.max_len])

            length = len(inds)

        x_cat = {
            c: torch.LongTensor(
                [self.X_cat[c][i] if self.X_cat[c][i] >= 0 else 0 for i in inds]
            )
            for c in self.cat_features
        }

        x_num = torch.FloatTensor(self.X_num[inds])
        y = torch.FloatTensor(self.y[inds]).reshape(-1)

        return x_cat, x_num, y, length


# def collate_fn(batch):
#     batch_size = len(batch)
#     lengths = [b[3] for b in batch]
#     max_len = max(lengths)

#     padded_x_cat = {key: [] for key in batch[0][0].keys()}
#     padded_x_num = []
#     padded_y = []

#     num_dim = batch[0][1].shape[1]

#     for x_cat, x_num, y, length in batch:
#         pad_len = max_len - length

#         for key in padded_x_cat:
#             val = x_cat[key]
#             if pad_len > 0:
#                 val = torch.cat([val, torch.zeros(pad_len, dtype=torch.long)])
#             padded_x_cat[key].append(val)

#         if pad_len > 0:
#             x_num = torch.cat([x_num, torch.zeros(pad_len, num_dim)])
#             y = torch.cat([y, torch.zeros(pad_len)])

#         padded_x_num.append(x_num)
#         padded_y.append(y)

#     padded_x_cat = {k: torch.stack(v) for k, v in padded_x_cat.items()}
#     padded_x_num = torch.stack(padded_x_num)
#     padded_y = torch.stack(padded_y)
#     lengths = torch.tensor(lengths, dtype=torch.long)

#     return padded_x_cat, padded_x_num, padded_y, lengths


class RankDataset(Dataset):
    def __init__(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        groups: pl.DataFrame,
        cat_features: list[str],
        num_features: list[str],
    ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.X_cat = {c: X[c].to_numpy() for c in cat_features}
        self.X_num = X[self.num_features].to_numpy()
        self.y = y.to_numpy()
        self.groups = groups["ranker_id"].to_numpy()

        self.group_to_indices = defaultdict(list)
        for i, g in enumerate(self.groups):
            self.group_to_indices[g].append(i)
        self.group_keys = list(self.group_to_indices.keys())

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        g = self.group_keys[idx]
        inds = self.group_to_indices[g]
        length = len(inds)

        x_cat = {
            c: torch.LongTensor(
                [self.X_cat[c][i] if self.X_cat[c][i] >= 0 else 0 for i in inds]
            )
            for c in self.cat_features
        }

        x_num = torch.FloatTensor(self.X_num[inds])
        y = torch.FloatTensor(self.y[inds]).reshape(-1)

        return x_cat, x_num, y, length


def collate_fn(batch):
    batch_size = len(batch)
    lengths = [b[3] for b in batch]
    max_len = max(lengths)

    padded_x_cat = {key: [] for key in batch[0][0].keys()}
    padded_x_num = []
    padded_y = []

    num_dim = batch[0][1].shape[1]

    for x_cat, x_num, y, length in batch:
        pad_len = max_len - length

        for key in padded_x_cat:
            val = x_cat[key]
            if pad_len > 0:
                val = torch.cat([val, torch.zeros(pad_len, dtype=torch.long)])
            padded_x_cat[key].append(val)

        if pad_len > 0:
            x_num = torch.cat([x_num, torch.zeros(pad_len, num_dim)])
            y = torch.cat([y, torch.zeros(pad_len)])

        padded_x_num.append(x_num)
        padded_y.append(y)

    padded_x_cat = {k: torch.stack(v) for k, v in padded_x_cat.items()}
    padded_x_num = torch.stack(padded_x_num)
    padded_y = torch.stack(padded_y)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_x_cat, padded_x_num, padded_y, lengths
