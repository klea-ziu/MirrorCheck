#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from mirrorcheck.encoders import load_open_clip_encoder
from mirrorcheck.similarity import cosine_similarity

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def read_pair_csv(path: Path, input_dir: Path | None, generated_dir: Path | None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Pair CSV must have a header.")

        possible_input_cols = ["input", "image", "original", "clean", "adv", "adversarial"]
        possible_generated_cols = ["generated", "gen", "reconstruction", "x_gen"]

        input_col = next((c for c in possible_input_cols if c in reader.fieldnames), None)
        generated_col = next((c for c in possible_generated_cols if c in reader.fieldnames), None)

        if input_col is None or generated_col is None:
            raise ValueError(
                "Pair CSV must contain one input column "
                f"{possible_input_cols} and one generated column {possible_generated_cols}."
            )

        for row in reader:
            inp = Path(row[input_col])
            gen = Path(row[generated_col])
            if not inp.is_absolute() and input_dir is not None:
                inp = input_dir / inp
            if not gen.is_absolute() and generated_dir is not None:
                gen = generated_dir / gen
            pairs.append((inp, gen))
    return pairs


def make_pairs(args) -> list[tuple[Path, Path]]:
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else None
    generated_dir = Path(args.generated_dir).expanduser() if args.generated_dir else None

    if args.pair_csv:
        return read_pair_csv(Path(args.pair_csv).expanduser(), input_dir, generated_dir)

    if input_dir is None or generated_dir is None:
        raise ValueError("--input-dir and --generated-dir are required unless --pair-csv is used.")

    inputs = image_files(input_dir, recursive=args.recursive)
    generated = image_files(generated_dir, recursive=args.recursive)

    if args.pairing == "name":
        generated_by_name = {p.name: p for p in generated}
        pairs = []
        missing = []
        for inp in inputs:
            gen = generated_by_name.get(inp.name)
            if gen is None:
                missing.append(inp.name)
            else:
                pairs.append((inp, gen))
        if missing and not args.quiet:
            print(f"Warning: {len(missing)} input images had no generated image with the same filename.")
            print("First missing examples:", ", ".join(missing[:5]))
        return pairs

    if args.pairing == "index":
        n = min(len(inputs), len(generated))
        if len(inputs) != len(generated) and not args.quiet:
            print(
                "Warning: input and generated folders have different numbers of images. "
                f"Pairing first {n} by sorted order. "
                f"inputs={len(inputs)}, generated={len(generated)}"
            )
        return list(zip(inputs[:n], generated[:n]))

    raise ValueError(f"Unknown pairing mode: {args.pairing}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MirrorCheck similarity for original/generated image pairs. "
            "Generated images may either share filenames with original images, "
            "be paired by sorted index, or be specified by a CSV mapping."
        )
    )
    parser.add_argument("--input-dir", default=None, help="Folder containing original input images.")
    parser.add_argument("--generated-dir", default=None, help="Folder containing regenerated images.")
    parser.add_argument(
        "--pairing",
        choices=["name", "index"],
        default="name",
        help=(
            "How to pair images when --pair-csv is not provided. "
            "Use 'name' when filenames match, or 'index' when folders are aligned by sorted order "
            "such as ILSVRC...png matching 00000.png, 00001.png, ..."
        ),
    )
    parser.add_argument(
        "--pair-csv",
        default=None,
        help=(
            "Optional CSV mapping pairs. Header should contain an input column "
            "such as input/image/original and a generated column such as generated/gen."
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Search for images recursively.")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--out-csv", default="mirrorcheck_scores.csv")
    parser.add_argument("--device", default=None, help="cuda, cpu, etc. Default: auto.")
    parser.add_argument("--encoder-model", default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--encoder-pretrained", default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of pairs to evaluate.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = make_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        raise RuntimeError("No image pairs found. Check paths and pairing mode.")

    encoder = load_open_clip_encoder(args.encoder_model, args.encoder_pretrained, device=args.device)

    rows = []
    for image_path, gen_path in tqdm(pairs, disable=args.quiet, desc="Evaluating pairs"):
        if not image_path.exists():
            if not args.quiet:
                print(f"Skipping missing input: {image_path}")
            continue
        if not gen_path.exists():
            if not args.quiet:
                print(f"Skipping missing generated image: {gen_path}")
            continue

        z_in = encoder.encode_image(str(image_path))
        z_gen = encoder.encode_image(str(gen_path))
        score = float(cosine_similarity(z_in, z_gen).item())
        rows.append(
            {
                "input": str(image_path),
                "generated": str(gen_path),
                "input_name": image_path.name,
                "generated_name": gen_path.name,
                "score": score,
                "threshold": args.threshold,
                "decision": "adversarial" if score < args.threshold else "clean",
            }
        )

    fieldnames = [
        "input",
        "generated",
        "input_name",
        "generated_name",
        "score",
        "threshold",
        "decision",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not args.quiet:
        print(f"Wrote {len(rows)} rows to {args.out_csv}")
        if rows:
            scores = [r["score"] for r in rows]
            print(f"Mean score: {sum(scores)/len(scores):.4f}")


if __name__ == "__main__":
    main()
