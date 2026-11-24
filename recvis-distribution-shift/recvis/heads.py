from dataclasses import dataclass

import torch
import torch.nn.functional as F

# CLIP's learned temperature, within a few percent across all five checkpoints.
LOGIT_SCALE = 100.0


@dataclass
class LinearHead:
    weight: torch.Tensor  # (feature_dim, num_classes)
    bias: torch.Tensor    # (num_classes,)

    def logits(self, features):
        return features @ self.weight + self.bias


def zero_shot_head(text_head, scale=LOGIT_SCALE):
    return LinearHead(scale * text_head, torch.zeros(text_head.shape[1]))


def train_linear_head(features, labels, init, *, epochs=20, batch_size=1024,
                      lr=1e-3, label_smoothing=0.1, device=None, seed=0):
    """Fit a linear head starting from `init` (LP-FT initialisation)."""
    device = device or features.device
    features, labels = features.to(device), labels.to(device)

    weight = init.weight.clone().to(device).requires_grad_(True)
    bias = init.bias.clone().to(device).requires_grad_(True)

    optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=0.0)
    steps = max(1, (len(features) + batch_size - 1) // batch_size) * epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        order = torch.randperm(len(features), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            loss = F.cross_entropy(
                features[index] @ weight + bias,
                labels[index],
                label_smoothing=label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()

    return LinearHead(weight.detach().cpu(), bias.detach().cpu())


def wise_ft(zero_shot, finetuned, alpha):
    """alpha=0 is pure zero-shot, alpha=1 pure probe."""
    return LinearHead(
        weight=(1 - alpha) * zero_shot.weight + alpha * finetuned.weight,
        bias=(1 - alpha) * zero_shot.bias + alpha * finetuned.bias,
    )


def accuracy(logits, labels):
    return float(logits.argmax(1).eq(labels).float().mean())
