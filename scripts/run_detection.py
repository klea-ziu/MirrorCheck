#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image

from mirrorcheck.detector import MirrorCheckDetector
from mirrorcheck.encoders import load_open_clip_encoder
from mirrorcheck.t2i import IdentityGenerator, StableDiffusionGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Run MirrorCheck on one image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--text", required=True, help="Victim model output text/caption/class description.")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--generated-image", default=None, help="Use an existing generated image instead of running T2I.")
    parser.add_argument("--sd-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out", default="generated.png")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = load_open_clip_encoder("ViT-B-32", "openai")

    if args.generated_image:
        generator = IdentityGenerator(args.generated_image)
    else:
        generator = StableDiffusionGenerator(args.sd_model)

    detector = MirrorCheckDetector(generator=generator, encoders=[encoder], threshold=args.threshold)
    result = detector.detect(args.image, text=args.text)
    result.generated_image.save(args.out)

    print(f"text: {result.text}")
    print(f"score: {result.score:.4f}")
    print(f"threshold: {result.threshold:.4f}")
    print(f"decision: {'adversarial' if result.is_adversarial else 'clean'}")
    print(f"generated image saved to: {args.out}")


if __name__ == "__main__":
    main()
