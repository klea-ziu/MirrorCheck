from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from PIL import Image


@dataclass
class StableDiffusionGenerator:
    """Text-to-image generator wrapper used by MirrorCheck."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    def __post_init__(self) -> None:
        try:
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError("Install diffusers and transformers to use StableDiffusionGenerator.") from exc

        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=guidance_scale or self.guidance_scale,
            generator=generator,
        ).images[0]
        return image


class IdentityGenerator:
    """Utility generator for tests/examples when generated images already exist."""

    def __init__(self, generated_image: Image.Image | str):
        self.generated_image = generated_image

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if isinstance(self.generated_image, str):
            return Image.open(self.generated_image).convert("RGB")
        return self.generated_image.convert("RGB")
