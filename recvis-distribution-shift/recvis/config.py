import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_var, *candidates):
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return next((c for c in candidates if c.exists()), candidates[0])


DATA_DIR = _resolve(
    "RECVIS_DATA_DIR",
    ROOT / "data",
    Path("/kaggle/input/mva-recvis-2026/recvis26_a3_public"),
    Path("/kaggle/input/mva-recvis-2026"),
)
OUTPUT_DIR = _resolve("RECVIS_OUTPUT_DIR", ROOT / "runs", Path("/kaggle/working"))
CACHE_DIR = _resolve("RECVIS_CACHE_DIR", ROOT / "cache", Path("/kaggle/working/cache"))

SPLIT_FILES = {"train": "train.npz", "validation": "validation.npz", "test": "test.npz"}
SAMPLE_SUBMISSION = "sample_submission.csv"

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CLASS_NAMES)

PHOTO_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low-resolution photo of a {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a black and white photo of a {}.",
)

RENDITION_TEMPLATES = (
    "a toy {}.",
    "a plastic toy {}.",
    "a plastic figurine of a {}.",
    "a wooden carving of a {}.",
    "a clay model of a {}.",
    "a plush toy {}.",
    "a lego {}.",
    "a lego model of a {}.",
    "an origami {}.",
    "a paper craft {}.",
    "a sculpture of a {}.",
    "a statue of a {}.",
    "a scale model of a {}.",
    "a miniature model of a {}.",
    "a die-cast model of a {}.",
    "a pencil sketch of a {}.",
    "a charcoal drawing of a {}.",
    "a line drawing of a {}.",
    "a cartoon {}.",
    "a painting of a {}.",
    "a watercolor painting of a {}.",
    "an oil painting of a {}.",
    "pixel art of a {}.",
    "a low-poly 3d render of a {}.",
    "a 3d render of a {}.",
    "an embroidered {}.",
    "a graffiti of a {}.",
    "a tattoo of a {}.",
    "a {} printed on a t-shirt.",
    "a drawing of a {} on a sign.",
    "a silhouette of a {}.",
    "a stained glass window of a {}.",
)

PROMPT_TEMPLATES = PHOTO_TEMPLATES + RENDITION_TEMPLATES
