from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from recvis import config

# (open_clip architecture, pretrained tag). One corpus each: DFN-5B, LAION-2B, WebLI, MetaCLIP-2 worldwide, WebLI at ViT-gopt scale.
REGISTRY = {
    "dfn_h_14":       ("ViT-H-14-quickgelu", "dfn5b"),
    "vit_h_14":       ("ViT-H-14", "laion2b_s32b_b79k"),
    "siglip2_so400m": ("ViT-SO400M-14-SigLIP2", "webli"),
    "metaclip2_h_14": ("ViT-H-14-worldwide-quickgelu", "metaclip2_worldwide"),
    "siglip2_g_16":   ("ViT-gopt-16-SigLIP2-256", "webli"),
}

ENSEMBLE = tuple(REGISTRY)


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    device: torch.device
    image_size: int
    mean: torch.Tensor = field(repr=False)
    std: torch.Tensor = field(repr=False)
    tokenizer: object = field(repr=False, default=None)

    @torch.inference_mode()
    def encode(self, pixels):
        return F.normalize(self.model.encode_image(pixels).float(), dim=-1)

    @torch.inference_mode()
    def zero_shot_head(self):
        """(feature_dim, num_classes) head from the prompt bank."""
        columns = []
        for class_name in config.CLASS_NAMES:
            prompts = [t.format(class_name) for t in config.PROMPT_TEMPLATES]
            tokens = self.tokenizer(prompts).to(self.device)
            embeddings = F.normalize(self.model.encode_text(tokens).float(), dim=-1)
            columns.append(F.normalize(embeddings.mean(0), dim=-1))
        return torch.stack(columns, dim=1)


def load_backbone(name, device=None, dtype=None):
    import open_clip

    architecture, tag = REGISTRY[name]
    device = device or default_device()
    # MPS and CPU handle fp16 poorly; CUDA does not.
    dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)

    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=tag, precision="fp32"
    )
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    image_size = model.visual.image_size
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    # SigLIP checkpoints use mean = std = 0.5, not the OpenAI constants.
    normalize = next(t for t in preprocess.transforms if hasattr(t, "mean"))
    shape = (1, 3, 1, 1)

    return Backbone(
        name=name,
        model=model,
        device=device,
        image_size=int(image_size),
        mean=torch.tensor(normalize.mean, device=device, dtype=dtype).view(shape),
        std=torch.tensor(normalize.std, device=device, dtype=dtype).view(shape),
        tokenizer=open_clip.get_tokenizer(architecture),
    )
