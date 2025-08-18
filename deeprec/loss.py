import torch
import torch.nn.functional as F


def ranknet_loss(scores, labels, lengths):
    batch_size = scores.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        l = lengths[i]
        s = scores[i][:l]
        y = labels[i][:l]

        diff_scores = s.unsqueeze(1) - s.unsqueeze(0)
        rel = y.unsqueeze(1) > y.unsqueeze(0)
        label_diff = rel.float()

        log_sigmoid = torch.nn.functional.logsigmoid(diff_scores)
        pair_loss = -label_diff * log_sigmoid

        total_loss += pair_loss.sum()

    return total_loss / batch_size


def xranknet_loss(scores, labels, lengths, temperature=1.0, margin=0.3, focus_topk=3):
    """
    xRankNet with emphasis on top-K items (based on labels, not scores).
    """
    batch_size = scores.size(0)
    total_loss = 0.0
    total_pairs = 0

    for i in range(batch_size):
        l = lengths[i]
        if l < 2:
            continue

        s = scores[i][:l]
        y = labels[i][:l]

        # Top-K mask based on ground truth labels
        topk = min(focus_topk, l)
        _, topk_idx = torch.topk(y, topk)
        topk_mask = torch.zeros_like(y, dtype=torch.bool)
        topk_mask[topk_idx] = True

        # Create pairwise difference matrices
        score_diff = s.unsqueeze(1) - s.unsqueeze(0)
        label_diff = y.unsqueeze(1) - y.unsqueeze(0)

        # valid pair: y_i > y_j and at least one of i, j is in top-K
        valid_pairs = (label_diff > 0) & (
            topk_mask.unsqueeze(1) | topk_mask.unsqueeze(0)
        )

        if valid_pairs.sum() == 0:
            continue

        importance = (label_diff.abs() * valid_pairs).float()

        if margin > 0.0:
            score_diff = score_diff - margin

        score_diff = score_diff / temperature
        pairwise_loss = -F.logsigmoid(score_diff) * importance

        total_loss += pairwise_loss.sum()
        total_pairs += importance.sum()

    if total_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return total_loss / total_pairs


def bpr_ranknet_loss(scores, labels, lengths, temperature=1.0, margin=0.0):
    batch_size = scores.size(0)
    total_loss = 0.0
    total_pairs = 0

    for i in range(batch_size):
        l = lengths[i]
        if l < 2:
            continue

        s = scores[i][:l]
        y = labels[i][:l]

        pos_mask = y > 0
        if pos_mask.sum() == 0:
            continue

        # Single pos sample
        pos_score = s[pos_mask][0]
        neg_scores = s[~pos_mask]

        # calculate score diff
        score_diff = pos_score - neg_scores  # [num_neg]
        if margin > 0.0:
            score_diff = score_diff - margin
        score_diff = score_diff / temperature

        # RankNet loss
        pairwise_loss = -F.logsigmoid(score_diff).sum()

        total_loss += pairwise_loss
        total_pairs += len(neg_scores)

    if total_pairs == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    return total_loss / total_pairs


def softmax_ce_loss(scores, labels, lengths):
    batch_size = scores.size(0)
    total_loss = 0.0
    total_groups = 0

    for i in range(batch_size):
        l = lengths[i]
        if l < 2:
            continue

        s = scores[i][:l]
        y = labels[i][:l]

        if y.sum() == 0:
            continue

        target = torch.argmax(y).long()
        loss = F.cross_entropy(s.unsqueeze(0), target.unsqueeze(0))  # [1, l] vs [1]
        total_loss += loss
        total_groups += 1

    return total_loss / max(1, total_groups)


def ranking_hinge_loss(scores, labels, lengths, margin=1.0):
    device = scores.device
    total_loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for i in range(scores.size(0)):  # batch size
        l = lengths[i]
        if l < 2:
            continue

        s = scores[i, :l]  # shape [l]
        y = labels[i, :l]  # one-hot, shape [l]

        if y.sum() == 0:
            continue

        pos_idx = torch.argmax(y)
        s_pos = s[pos_idx]  # scalar
        s_neg = s[torch.arange(l, device=device) != pos_idx]  # shape [l-1]

        # vectorized hinge loss: max(0, margin - (s_pos - s_neg))
        losses = F.relu(margin - (s_pos - s_neg))  # shape [l-1]
        total_loss += losses.sum()
        total_pairs += losses.numel()

    return total_loss / max(1, total_pairs)
