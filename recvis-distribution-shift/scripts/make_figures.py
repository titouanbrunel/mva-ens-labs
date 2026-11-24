"""Regenerate the README figures in assets/.

    python scripts/make_figures.py --predictions runs/submission.csv

Needs the competition NPZ files (RECVIS_DATA_DIR) and the cached dfn_h_14
features (RECVIS_CACHE_DIR), both produced by main.py. The accuracy numbers in
the two result figures are copied from the Kaggle history and the report.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recvis import config  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SOURCE = "#2a78d6"   # categorical slot 1
TARGET = "#eb6834"   # categorical slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(split):
    return np.load(Path(config.DATA_DIR) / config.SPLIT_FILES[split])


def samples(predictions, rng):
    """One CIFAR-10 photo per class, then three target images the model put in that class."""
    train = load("train")
    test = load("test")
    pred = pd.read_csv(predictions).set_index("id")["label"]
    test_pred = pred.loc[test["ids"]].to_numpy()

    rows = 4
    fig, axes = plt.subplots(rows, 10, figsize=(10, rows * 1.08 + 0.5))
    for k, name in enumerate(config.CLASS_NAMES):
        picks = [train["images"][rng.choice(np.flatnonzero(train["labels"] == k))]]
        picks += list(test["images"][rng.choice(np.flatnonzero(test_pred == k), rows - 1, replace=False)])
        for r, img in enumerate(picks):
            ax = axes[r, k]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(SOURCE if r == 0 else TARGET)
                s.set_linewidth(2)
        axes[0, k].set_title(name, fontsize=9, fontweight="normal", loc="center", color=INK_2)
    axes[0, 0].set_ylabel("CIFAR-10\n(train)", fontsize=9, rotation=0, ha="right", va="center", color=SOURCE)
    axes[2, 0].set_ylabel("target\n(test)", fontsize=9, rotation=0, ha="right", va="center", color=TARGET)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.01, wspace=0.08, hspace=0.08)
    fig.savefig(ASSETS / "samples.png", dpi=160)
    plt.close(fig)


def feature_space(backbone, rng, n=3000):
    """t-SNE of frozen features, source vs target, and how separable the two are."""
    cache = Path(config.CACHE_DIR)
    src = np.load(cache / f"{backbone}__train_flip.npy", mmap_mode="r")
    tgt = np.load(cache / f"{backbone}__test_flip.npy", mmap_mode="r")
    src = np.asarray(src[np.sort(rng.choice(len(src), n, replace=False))], dtype=np.float32)
    tgt = np.asarray(tgt[np.sort(rng.choice(len(tgt), n, replace=False))], dtype=np.float32)
    x = np.concatenate([src, tgt])
    y = np.r_[np.zeros(n), np.ones(n)]

    # Domain classifier: if a linear probe tells source from target, the shift is real.
    sep = cross_val_score(LogisticRegression(max_iter=2000, C=1.0), x, y, cv=5).mean()

    emb = TSNE(perplexity=40, init="pca", random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(x))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    order = rng.permutation(len(x))
    colors = np.where(y[order] == 0, SOURCE, TARGET)
    ax.scatter(emb[order, 0], emb[order, 1], s=5, c=colors, linewidths=0, alpha=0.7)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Same ten classes, two clouds ({backbone} features, t-SNE)")
    ax.scatter([], [], s=30, c=SOURCE, label="CIFAR-10 train")
    ax.scatter([], [], s=30, c=TARGET, label="target test")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(ASSETS / "feature_space.png", dpi=160)
    plt.close(fig)
    return sep


def validation_vs_target():
    """Two panels, one message: source validation does not rank what transfers."""
    ladder = [  # (label, validation, target public LB)
        ("ViT-B/32", 0.9460, 0.89369),
        ("ViT-H/14 DFN", 0.9879, 0.94159),
        ("5 encoders", 0.9917, 0.95516),
        ("5 encoders\n+ DomainNet in train", 0.9945, 0.94224),
    ]
    instrument = [  # vit_b_32, (recipe, validation, DomainNet held-out macro)
        ("ep80 · +DomainNet · α .4", 0.9575, 0.8468),
        ("ep20 · α .4 (shipped)", 0.9445, 0.8369),
        ("ep80 · α .4", 0.9584, 0.8344),
        ("ep80 · α 1.0", 0.9703, 0.8130),
    ]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax in (a, b):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    v, t = zip(*[(r[1], r[2]) for r in ladder])
    a.plot(v[:3], t[:3], color=SOURCE, linewidth=2, zorder=2)
    a.annotate("", xy=(v[3], t[3]), xytext=(v[2], t[2]),
               arrowprops=dict(arrowstyle="-|>", color=TARGET, linewidth=2, shrinkA=6, shrinkB=6))
    a.scatter(v[:3], t[:3], s=60, color=SOURCE, edgecolor=SURFACE, linewidth=2, zorder=3)
    a.scatter(v[3:], t[3:], s=60, color=TARGET, edgecolor=SURFACE, linewidth=2, zorder=3)
    offsets = [(8, -4), (-10, -2), (-10, -2), (9, -2)]
    for (label, x, y), off in zip(ladder, offsets):
        a.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=9, color=INK_2,
                   ha="right" if off[0] < 0 else "left", va="center")
    a.set_xlabel("validation accuracy (CIFAR-10 photos)")
    a.set_ylabel("target accuracy (Kaggle public)")
    a.set_title("Better encoder: both go up.\nMore training data: validation up, target down.")
    a.set_xlim(0.940, 1.013)
    a.set_xticks(np.arange(0.94, 0.991, 0.01))

    v, t = zip(*[(r[1], r[2]) for r in instrument])
    b.scatter(v, t, s=60, color=SOURCE, edgecolor=SURFACE, linewidth=2, zorder=3)
    b.scatter([v[-1]], [t[-1]], s=60, color=TARGET, edgecolor=SURFACE, linewidth=2, zorder=4)
    offsets = [(-8, 6), (8, -4), (8, -10), (-8, 8)]
    for (label, x, y), off in zip(instrument, offsets):
        b.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=9, color=INK_2,
                   ha="right" if off[0] < 0 else "left")
    b.set_xlabel("validation accuracy (CIFAR-10 photos)")
    b.set_ylabel("held-out DomainNet domain (macro)")
    b.set_title("Same ViT-B/32 probe, four recipes:\nthe best on validation is the worst off-domain.")
    b.set_xlim(0.938, 0.976)

    fig.tight_layout(w_pad=3)
    fig.savefig(ASSETS / "validation_vs_target.png", dpi=160)
    plt.close(fig)


def class_prior():
    """Predicted class balance on the target: the DomainNet head drifts away from uniform."""
    cifar = [0.099, 0.0926, 0.097, 0.1002, 0.0995, 0.1008, 0.0999, 0.098, 0.0994, 0.1137]
    dnet = [0.0985, 0.0785, 0.0982, 0.1013, 0.0985, 0.0987, 0.0996, 0.0973, 0.1004, 0.1291]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axhline(0.1, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(-0.45, 0.1008, "uniform", color=INK_2, fontsize=9, va="bottom")
    ax.vlines(x, np.minimum(cifar, dnet), np.maximum(cifar, dnet), color=GRID, linewidth=2, zorder=2)
    ax.scatter(x, cifar, s=64, color=SOURCE, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="head fit on CIFAR-10 (target 0.955)")
    ax.scatter(x, dnet, s=64, color=TARGET, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="head fit on CIFAR-10 + DomainNet (target 0.942)")
    ax.set_xticks(x, config.CLASS_NAMES)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0.074, 0.134)
    ax.set_ylabel("share of target predictions")
    ax.yaxis.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    ax.set_title("The target is balanced. Predictions drift from automobile to truck.")
    fig.tight_layout()
    fig.savefig(ASSETS / "class_prior.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="a submission CSV, used only to group target thumbnails by class")
    parser.add_argument("--backbone", default="dfn_h_14")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    samples(args.predictions, rng)
    sep = feature_space(args.backbone, rng)
    print(f"linear domain classifier, source vs target, 5-fold accuracy: {sep:.3f}")
    validation_vs_target()
    class_prior()
    print(f"wrote {sorted(p.name for p in ASSETS.glob('*.png'))}")


if __name__ == "__main__":
    main()
