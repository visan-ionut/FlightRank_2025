import torch


def hitrate_at_k(scores, labels, lengths, k=3, min_group_size=10):
    batch_size = scores.size(0)
    hits = 0.0
    valid_count = 0
    for i in range(batch_size):
        l = lengths[i]
        if l < min_group_size:
            continue
        length = min(l, k)
        s = scores[i][:l]
        y = labels[i][:l]
        _, topk_idx = torch.topk(s, length)
        topk_labels = y[topk_idx]
        hits += (topk_labels > 0).any().float().item()
        valid_count += 1
    return hits / valid_count if valid_count > 0 else 0.0


def ndcg_at_k(scores, labels, lengths, k=3, min_group_size=10):
    batch_size = scores.size(0)
    total_ndcg = 0.0
    valid_count = 0

    for i in range(batch_size):
        l = lengths[i]
        if l < min_group_size:
            continue

        s = scores[i][:l]
        y = labels[i][:l]

        _, idx_pred = torch.topk(s, min(k, l))
        _, idx_ideal = torch.topk(y, min(k, l))

        dcg = (
            y[idx_pred]
            / torch.log2(torch.arange(2, 2 + len(idx_pred), device=y.device).float())
        ).sum()
        idcg = (
            y[idx_ideal]
            / torch.log2(torch.arange(2, 2 + len(idx_ideal), device=y.device).float())
        ).sum()

        if idcg > 0:
            ndcg = (dcg / idcg).item()
        else:
            ndcg = 0.0

        total_ndcg += ndcg
        valid_count += 1

    return total_ndcg / valid_count if valid_count > 0 else 0.0


def map_at_k(scores, labels, lengths, k=3, min_group_size=10):
    batch_size = scores.size(0)
    total_ap = 0.0
    valid_count = 0

    for i in range(batch_size):
        l = lengths[i]
        if l < min_group_size:
            continue

        s = scores[i][:l]
        y = labels[i][:l]

        _, idx_pred = torch.topk(s, min(k, l))
        y_true = y[idx_pred] > 0

        if y_true.sum() == 0:
            continue

        precisions = [
            (y_true[: j + 1].float().sum() / (j + 1))
            for j in range(len(y_true))
            if y_true[j]
        ]
        if len(precisions) == 0:
            continue

        ap = torch.stack(precisions).mean().item()
        total_ap += ap
        valid_count += 1

    return total_ap / valid_count if valid_count > 0 else 0.0
