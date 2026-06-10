from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from typing import Callable, Optional, Sequence
from PIL import Image
import torch

from .detector import DetectionResult
from .similarity import cosine_similarity, mean_similarity


@contextmanager
def one_time_parameter_noise(model: torch.nn.Module, noise_scale: float):
    """Temporarily add Gaussian OTU noise to model parameters, then restore."""
    if noise_scale <= 0:
        yield
        return

    originals = []
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            originals.append((param, param.detach().clone()))
            param.add_(torch.randn_like(param) * noise_scale)
    try:
        yield
    finally:
        with torch.no_grad():
            for param, old_value in originals:
                param.copy_(old_value)


class StochasticMirrorCheckDetector:
    """Stochastic MirrorCheck detector.

    At inference time, randomly selects a T2I generator, randomly samples an
    encoder subset, and applies one-time-use parameter perturbations before
    computing the ensemble similarity score.
    """

    def __init__(
        self,
        generators: Sequence,
        encoders: Sequence,
        threshold: float,
        n_encoders: int = 1,
        otu_noise_scale: float = 5e-6,
        seed: Optional[int] = None,
    ):
        if not generators:
            raise ValueError("At least one T2I generator is required.")
        if not encoders:
            raise ValueError("At least one image encoder is required.")
        if n_encoders < 1:
            raise ValueError("n_encoders must be >= 1.")
        self.generators = list(generators)
        self.encoders = list(encoders)
        self.threshold = float(threshold)
        self.n_encoders = min(int(n_encoders), len(self.encoders))
        self.otu_noise_scale = float(otu_noise_scale)
        self.rng = random.Random(seed)

    def detect(
        self,
        image: Image.Image | str,
        text: Optional[str] = None,
        victim_fn: Optional[Callable[[Image.Image | str], str]] = None,
        seed: Optional[int] = None,
    ) -> DetectionResult:
        if text is None:
            if victim_fn is None:
                raise ValueError("Provide either `text` or `victim_fn`.")
            text = victim_fn(image)

        generator = self.rng.choice(self.generators)
        generated = generator.generate(text, seed=seed)
        selected_encoders = self.rng.sample(self.encoders, k=self.n_encoders)

        per_encoder_scores: dict[str, float] = {}
        for encoder in selected_encoders:
            with one_time_parameter_noise(encoder.model, self.otu_noise_scale):
                z_in = encoder.encode_image(image)
                z_gen = encoder.encode_image(generated)
            per_encoder_scores[encoder.name] = float(cosine_similarity(z_in, z_gen).item())

        score = mean_similarity(per_encoder_scores.values())
        return DetectionResult(
            score=score,
            threshold=self.threshold,
            is_adversarial=score < self.threshold,
            text=text,
            generated_image=generated,
            per_encoder_scores=per_encoder_scores,
        )
