"""Final submission: linear probes over five frozen CLIP encoders.

    python main.py                 # extract features if absent, then predict
    python main.py --cached-only   # fail rather than download any checkpoint

Features are cached under RECVIS_CACHE_DIR, so a second run costs seconds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import recvis  # noqa: F401  imports numpy before torch
from recvis import backbones, config, features, heads

EPOCHS = 20
LR = 1e-3
LABEL_SMOOTHING = 0.1
ALPHA = 0.4


def member_log_probs(name, cached_only):
    """Log-probabilities on validation and test for one encoder."""
    source = name if cached_only or features.cache_path(name, "test").is_file() \
        else backbones.load_backbone(name)

    train_x, train_y, _ = features.get_features(source, "train")
    val_x, val_y, _ = features.get_features(source, "validation")
    test_x, _, test_ids = features.get_features(source, "test")

    zero_shot = heads.zero_shot_head(features.get_text_head(source))
    probe = heads.train_linear_head(
        train_x, train_y, zero_shot,
        epochs=EPOCHS, lr=LR, label_smoothing=LABEL_SMOOTHING,
    )
    head = heads.wise_ft(zero_shot, probe, ALPHA)

    print(f"{name:16s} val {heads.accuracy(head.logits(val_x), val_y):.4f}")
    return (F.log_softmax(head.logits(val_x), dim=1),
            F.log_softmax(head.logits(test_x), dim=1),
            val_y, test_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", nargs="+", default=list(backbones.ENSEMBLE))
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(config.OUTPUT_DIR) / "submission.csv")
    args = parser.parse_args()

    val_stack, test_stack = [], []
    for name in args.members:
        val_lp, test_lp, val_y, test_ids = member_log_probs(name, args.cached_only)
        val_stack.append(val_lp)
        test_stack.append(test_lp)

    # Product of experts: the mean of the log-probabilities.
    val_pooled = torch.stack(val_stack).mean(0)
    test_pooled = torch.stack(test_stack).mean(0)
    accuracy = heads.accuracy(val_pooled, val_y)
    print(f"{'ensemble':16s} val {accuracy:.4f}")

    predictions = test_pooled.argmax(1).numpy()
    write_submission(test_ids.numpy(), predictions, args.output, accuracy, args.members)


def write_submission(ids, predictions, output_path, val_accuracy, members):
    sample = pd.read_csv(Path(config.DATA_DIR) / config.SAMPLE_SUBMISSION)
    lookup = pd.Series(predictions, index=ids)
    missing = sample["id"][~sample["id"].isin(lookup.index)]
    if len(missing):
        raise ValueError(f"{len(missing)} sample ids missing, e.g. {missing.iloc[0]}.")

    frame = pd.DataFrame({"id": sample["id"], "label": lookup.loc[sample["id"]].to_numpy()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    balance = np.bincount(frame["label"], minlength=config.NUM_CLASSES) / len(frame)
    output_path.with_suffix(".json").write_text(json.dumps({
        "members": members,
        "epochs": EPOCHS, "lr": LR, "label_smoothing": LABEL_SMOOTHING, "alpha": ALPHA,
        "validation_accuracy": val_accuracy,
        "class_balance": [round(float(b), 4) for b in balance],
    }, indent=2))
    print(f"wrote {output_path}  rows={len(frame)}")


if __name__ == "__main__":
    main()
