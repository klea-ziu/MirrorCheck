from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from PIL import Image

from .similarity import cosine_similarity, mean_similarity


@dataclass
class DetectionResult:
    score: float
    threshold: float
    is_adversarial: bool
    text: str
    generated_image: Image.Image
    per_encoder_scores: dict[str, float]


class MirrorCheckDetector:
    """Vanilla MirrorCheck detector.

    The detector compares the input image with a T2I reconstruction of the
    victim model's textual interpretation. Low similarity is flagged as adversarial.
    """

    def __init__(self, generator, encoders: Sequence, threshold: float):
        if not encoders:
            raise ValueError("At least one image encoder is required.")
        self.generator = generator
        self.encoders = list(encoders)
        self.threshold = float(threshold)

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

        generated = self.generator.generate(text, seed=seed)
        per_encoder_scores: dict[str, float] = {}

        for encoder in self.encoders:
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
