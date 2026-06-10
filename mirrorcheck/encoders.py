from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from PIL import Image
import torch


@dataclass
class ImageEncoder:
    """Small wrapper exposing a common encode_image API."""

    name: str
    model: torch.nn.Module
    preprocess: Callable
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image | str) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).detach().cpu()


def default_open_clip_pretrained(model_name: str) -> str:
    """Legacy defaults used in the original MirrorCheck experiment notebook."""
    return "laion400m_e31" if "ViT" in model_name else "yfcc15m"


def load_open_clip_encoder(
    model_name: str = "ViT-B-32",
    pretrained: Optional[str] = None,
    device: Optional[str] = None,
) -> ImageEncoder:
    """Load an OpenCLIP image encoder.

    Examples:
        model_name='ViT-B-32', pretrained='openai'
        model_name='ViT-B-16', pretrained='laion400m_e31'
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch to use OpenCLIP encoders.") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = pretrained or default_open_clip_pretrained(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    return ImageEncoder(name=f"open_clip/{model_name}/{pretrained}", model=model, preprocess=preprocess, device=device)


def load_clip_encoder(name: str = "ViT-B/32", device: Optional[str] = None) -> ImageEncoder:
    """Load OpenAI CLIP, when the `clip` package is installed."""
    try:
        import clip
    except ImportError as exc:
        raise ImportError("Install the OpenAI CLIP package to use load_clip_encoder.") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(name, device=device)
    return ImageEncoder(name=f"clip/{name}", model=model, preprocess=preprocess, device=device)


def load_encoder(
    model_name: str,
    library: str = "open_clip",
    pretrained: Optional[str] = None,
    device: Optional[str] = None,
) -> ImageEncoder:
    """Load an encoder from either OpenAI CLIP or OpenCLIP.

    library may be:
      - 'clip' for the OpenAI CLIP package, using names like 'RN50' or 'ViT-B/32'
      - 'open_clip' for open_clip_torch, using names like 'RN50' or 'ViT-B-32'
      - 'openai' as a convenience alias for open_clip pretrained='openai'
    """
    lib = library.lower().strip()
    if lib == "clip":
        return load_clip_encoder(name=model_name, device=device)
    if lib == "openai":
        return load_open_clip_encoder(model_name=model_name, pretrained="openai", device=device)
    if lib == "open_clip":
        return load_open_clip_encoder(model_name=model_name, pretrained=pretrained, device=device)
    raise ValueError(f"Unknown encoder library '{library}'. Use 'clip', 'open_clip', or 'openai'.")
