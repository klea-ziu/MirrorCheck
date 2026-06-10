#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirrorcheck.encoders import load_encoder
from mirrorcheck.similarity import cosine_similarity
from mirrorcheck.stochastic import one_time_parameter_noise

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LEGACY_10_ENCODERS = (
    "RN50:clip,"
    "RN101:clip,"
    "ViT-B/16:clip,"
    "ViT-B/32:clip,"
    "ViT-L/14:clip,"
    "RN50:open_clip:yfcc15m,"
    "RN101:open_clip:yfcc15m,"
    "ViT-B-16:open_clip:laion400m_e31,"
    "ViT-B-32:open_clip:laion400m_e31,"
    "ViT-L-14:open_clip:laion400m_e31"
)

OPENCLIP_OPENAI_5_ENCODERS = (
    "RN50:openai,"
    "RN101:openai,"
    "ViT-B-16:openai,"
    "ViT-B-32:openai,"
    "ViT-L-14:openai"
)


def parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def resolve_encoder_preset(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name == "legacy_10":
        return LEGACY_10_ENCODERS
    if name == "openclip_openai_5":
        return OPENCLIP_OPENAI_5_ENCODERS
    raise ValueError(f"Unknown encoder preset: {name}")


def parse_encoders(s: str) -> list[tuple[str, str, Optional[str]]]:
    """Parse encoder specs.

    Accepted forms:
      model:clip
      model:openai                     -> open_clip with pretrained='openai'
      model:open_clip                  -> open_clip with legacy default pretrained
      model:open_clip:pretrained_name

    Examples:
      RN50:clip
      ViT-B/32:clip
      ViT-B-16:open_clip:laion400m_e31
      ViT-B-32:openai
    """
    out: list[tuple[str, str, Optional[str]]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 1:
            model, library, pretrained = parts[0], "openai", None
        elif len(parts) == 2:
            model, library = parts
            pretrained = None
        elif len(parts) == 3:
            model, library, pretrained = parts
        else:
            raise ValueError(f"Invalid encoder spec '{item}'. Use model:library[:pretrained].")
        out.append((model.strip(), library.strip(), pretrained.strip() if pretrained else None))
    return out


def image_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def pairs_by_index(input_dir: Path, generated_dir: Path) -> list[tuple[Path, Path]]:
    inputs = image_files(input_dir)
    generated = image_files(generated_dir)
    n = min(len(inputs), len(generated))
    if len(inputs) != len(generated):
        print(f"Warning: {input_dir} has {len(inputs)} images, {generated_dir} has {len(generated)} images. Using first {n} sorted pairs.")
    return list(zip(inputs[:n], generated[:n]))


def pairs_by_name(input_dir: Path, generated_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for p in image_files(input_dir):
        g = generated_dir / p.name
        if g.exists():
            pairs.append((p, g))
        else:
            print(f"Skipping {p.name}: no generated file with same name.")
    return pairs


def pairs_by_csv(mapping_csv: Path, root: Path | None = None) -> list[tuple[Path, Path, int]]:
    """Mapping CSV columns: input, generated, label. label: 0 clean, 1 adversarial."""
    rows = []
    with open(mapping_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"input", "generated", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV must contain columns {required}; missing {missing}")
        for r in reader:
            inp = Path(r["input"])
            gen = Path(r["generated"])
            if root is not None:
                if not inp.is_absolute():
                    inp = root / inp
                if not gen.is_absolute():
                    gen = root / gen
            rows.append((inp, gen, int(r["label"])))
    return rows


def build_labeled_pairs(args) -> list[dict]:
    if args.mapping_csv:
        rows = pairs_by_csv(Path(args.mapping_csv), Path(args.mapping_root) if args.mapping_root else None)
        pairs = [{"input": i, "generated": g, "label": y} for i, g, y in rows]
    else:
        clean_dir = Path(args.clean_dir)
        clean_generated_dir = Path(args.clean_generated_dir)
        adv_dir = Path(args.adv_dir)
        adv_generated_dir = Path(args.adv_generated_dir)
        pair_fn = pairs_by_index if args.pairing == "index" else pairs_by_name
        clean_pairs = pair_fn(clean_dir, clean_generated_dir)
        adv_pairs = pair_fn(adv_dir, adv_generated_dir)
        pairs = []
        pairs.extend({"input": i, "generated": g, "label": 0} for i, g in clean_pairs)
        pairs.extend({"input": i, "generated": g, "label": 1} for i, g in adv_pairs)

    if args.max_samples is not None:
        clean = [p for p in pairs if p["label"] == 0][: args.max_samples]
        adv = [p for p in pairs if p["label"] == 1][: args.max_samples]
        pairs = clean + adv
    return pairs


def load_encoder_zoo(encoder_specs: list[tuple[str, str, Optional[str]]], device: str):
    encoders = []
    for model_name, library, pretrained in encoder_specs:
        spec = f"{model_name}:{library}" + (f":{pretrained}" if pretrained else "")
        print(f"Loading encoder: {spec}")
        encoders.append(load_encoder(model_name=model_name, library=library, pretrained=pretrained, device=device))
    return encoders


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_pair(input_path: Path, generated_path: Path, encoder) -> float:
    z_in = encoder.encode_image(str(input_path))
    z_gen = encoder.encode_image(str(generated_path))
    return float(cosine_similarity(z_in, z_gen).item())


def select_encoders_config_mode(encoders, n_encoders: int, rng: random.Random):
    """Legacy notebook-style sampling: shuffle then sample once per configuration."""
    if n_encoders > len(encoders):
        raise ValueError(f"Requested {n_encoders} encoders, but only {len(encoders)} are available.")
    pool = list(encoders)
    rng.shuffle(pool)
    return rng.sample(pool, n_encoders)


def compute_config_scores_config_mode(pairs, selected_encoders, noise_scale: float, otu_mode: str):
    """Compute ensemble scores using a fixed selected encoder subset per config.

    This matches the old experiment notebook: select encoders once for a configuration,
    add OTU noise once to each selected model, score all image pairs, then average scores
    across selected encoders.
    """
    all_encoder_scores = []
    for enc in selected_encoders:
        if otu_mode == "legacy_once":
            context = one_time_parameter_noise(enc.model, noise_scale)
        elif otu_mode == "none":
            context = torch.no_grad()  # dummy context
        else:
            context = None

        scores = []
        if context is None:
            # per_image: new OTU noise is sampled for each image pair.
            for item in tqdm(pairs, leave=False):
                with one_time_parameter_noise(enc.model, noise_scale):
                    scores.append(score_pair(item["input"], item["generated"], enc))
        else:
            with context:
                for item in tqdm(pairs, leave=False):
                    scores.append(score_pair(item["input"], item["generated"], enc))
        all_encoder_scores.append(scores)

    return np.mean(np.asarray(all_encoder_scores, dtype=float), axis=0)


def compute_config_scores_per_image_mode(pairs, encoders, n_encoders: int, noise_scale: float, seed: int, num_runs: int):
    """Fully stochastic mode: random encoder subset per image and run."""
    scores = []
    for idx, item in enumerate(tqdm(pairs)):
        run_scores = []
        rng = random.Random(seed + idx)
        for _run in range(num_runs):
            selected = rng.sample(encoders, k=n_encoders)
            per_encoder = []
            for enc in selected:
                with one_time_parameter_noise(enc.model, noise_scale):
                    per_encoder.append(score_pair(item["input"], item["generated"], enc))
            run_scores.append(float(np.mean(per_encoder)))
        scores.append(float(np.mean(run_scores)))
    return np.asarray(scores, dtype=float)


def roc_arrays(scores: np.ndarray, labels: np.ndarray):
    # In the old notebook, clean=1, adversarial=0 and y_score is similarity.
    try:
        from sklearn.metrics import roc_curve
        y_true = (labels == 0).astype(int)  # clean positive for ROC, same as old notebook
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        return fpr, tpr, thresholds
    except Exception:
        thresholds = np.unique(scores)
        thresholds = np.concatenate(([scores.max() + 1e-6], thresholds[::-1], [scores.min() - 1e-6]))
        fpr_list, tpr_list = [], []
        y_clean = labels == 0
        for tau in thresholds:
            pred_clean = scores >= tau
            tp = np.logical_and(pred_clean, y_clean).sum()
            fn = np.logical_and(~pred_clean, y_clean).sum()
            fp = np.logical_and(pred_clean, ~y_clean).sum()
            tn = np.logical_and(~pred_clean, ~y_clean).sum()
            tpr_list.append(tp / (tp + fn) if (tp + fn) else 0.0)
            fpr_list.append(fp / (fp + tn) if (fp + tn) else 0.0)
        return np.asarray(fpr_list), np.asarray(tpr_list), np.asarray(thresholds)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, mode: str) -> tuple[float, float]:
    """Choose threshold. Decision is adversarial if similarity < threshold.

    mode='youden': maximize TPR_adv - FPR_adv.
    mode='equal_tpr_tnr': legacy notebook rule, choose ROC point where clean TPR ~= clean TNR.
    Returns threshold and AUC where AUC is computed using clean-positive ROC as in old notebook.
    """
    fpr_clean, tpr_clean, thresholds = roc_arrays(scores, labels)
    auc = float(np.trapz(tpr_clean, fpr_clean))

    if mode == "equal_tpr_tnr":
        idx = int(np.argmin(np.abs(tpr_clean - (1.0 - fpr_clean))))
        return float(thresholds[idx]), auc

    if mode == "youden":
        candidates = np.unique(scores)
        candidates = np.concatenate(([scores.min() - 1e-6], candidates, [scores.max() + 1e-6]))
        best_tau = float(candidates[0])
        best_j = -1e9
        for tau in candidates:
            metrics = compute_metrics(scores, labels, tau)
            j = metrics["tpr"] - metrics["fpr"]
            if j > best_j:
                best_j = j
                best_tau = float(tau)
        return best_tau, auc

    raise ValueError(f"Unknown threshold mode: {mode}")


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    # label 1 = adversarial. Decision: adversarial if score < threshold.
    pred = scores < threshold
    y = labels.astype(bool)
    tp = int(np.logical_and(pred, y).sum())
    fn = int(np.logical_and(~pred, y).sum())
    fp = int(np.logical_and(pred, ~y).sum())
    tn = int(np.logical_and(~pred, ~y).sum())
    total = len(labels)
    acc = (tp + tn) / total if total else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(precision),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_total": int(total),
        "n_clean": int((labels == 0).sum()),
        "n_adversarial": int((labels == 1).sum()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stochastic MirrorCheck on clean/adversarial original-generated image pairs.")

    parser.add_argument("--clean-dir", help="Folder containing original clean images.")
    parser.add_argument("--clean-generated-dir", help="Folder containing generated images for clean images.")
    parser.add_argument("--adv-dir", help="Folder containing original adversarial images.")
    parser.add_argument("--adv-generated-dir", help="Folder containing generated images for adversarial images.")
    parser.add_argument("--mapping-csv", help="Optional CSV with columns: input,generated,label. label 0=clean, 1=adversarial.")
    parser.add_argument("--mapping-root", help="Optional root for relative paths in mapping CSV.")
    parser.add_argument("--pairing", choices=["index", "name"], default="index", help="How to pair images when using folders. Use index when filenames differ but sorted order matches.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max clean samples and max adversarial samples. Use 100 to reproduce the old notebook snippet.")

    parser.add_argument("--encoder-preset", choices=["legacy_10", "openclip_openai_5"], default=None, help="Use a predefined encoder zoo. legacy_10 matches the old notebook.")
    parser.add_argument("--encoders", default=OPENCLIP_OPENAI_5_ENCODERS, help="Comma-separated encoder specs. Examples: 'RN50:clip,ViT-B/32:clip,ViT-B-16:open_clip:laion400m_e31'.")
    parser.add_argument("--n-encoders", default="5", help="Comma-separated values, e.g. '1,3,5,7,10'.")
    parser.add_argument("--otu-noise-scale", default="5e-6", help="Comma-separated values, e.g. '5e-6,5e-4,1e-3'.")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of stochastic repetitions. Used in per-image sampling mode.")
    parser.add_argument("--sampling-mode", choices=["config", "per_image"], default="config", help="config matches the old notebook: sample encoder subset once per configuration. per_image resamples per image/run.")
    parser.add_argument("--otu-mode", choices=["legacy_once", "per_image", "none"], default="legacy_once", help="legacy_once matches old notebook: add noise once per selected model/config. per_image resamples OTU noise per image.")
    parser.add_argument("--threshold", default="auto", help="Use 'auto' or pass a numeric threshold.")
    parser.add_argument("--threshold-mode", choices=["equal_tpr_tnr", "youden"], default="equal_tpr_tnr", help="equal_tpr_tnr matches the old notebook. youden maximizes TPR-FPR for adversarial detection.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--out-csv", default="stochastic_scores.csv")
    parser.add_argument("--summary-csv", default="stochastic_summary.csv")
    parser.add_argument("--summary-json", default="stochastic_summary.json")
    args = parser.parse_args()

    if not args.mapping_csv:
        required = [args.clean_dir, args.clean_generated_dir, args.adv_dir, args.adv_generated_dir]
        if any(v is None for v in required):
            parser.error("Provide either --mapping-csv or all of --clean-dir, --clean-generated-dir, --adv-dir, --adv-generated-dir.")
    if args.sampling_mode == "per_image" and args.otu_mode == "legacy_once":
        parser.error("--otu-mode legacy_once only makes sense with --sampling-mode config. Use --otu-mode per_image or --sampling-mode config.")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    pairs = build_labeled_pairs(args)
    if not pairs:
        raise RuntimeError("No image pairs found.")

    encoder_string = resolve_encoder_preset(args.encoder_preset) or args.encoders
    encoder_specs = parse_encoders(encoder_string)
    encoders = load_encoder_zoo(encoder_specs, args.device)
    n_values = parse_list_int(args.n_encoders)
    noise_values = parse_list_float(args.otu_noise_scale)
    for n in n_values:
        if n > len(encoders):
            raise ValueError(f"Requested n_encoders={n}, but encoder zoo has only {len(encoders)} encoders.")

    labels = np.asarray([int(p["label"]) for p in pairs], dtype=int)

    all_score_rows = []
    summary_rows = []

    # One RNG that advances across configurations to mimic the old notebook's loop.
    config_rng = random.Random(args.seed)

    for noise_scale in noise_values:
        print(f"\nRunning noise scale {noise_scale} with seed {args.seed}")
        for n_encoders in n_values:
            print(f"\nEvaluating n_encoders={n_encoders}, otu_noise_scale={noise_scale}")

            if args.sampling_mode == "config":
                selected = select_encoders_config_mode(encoders, n_encoders, config_rng)
                print("Selected encoders:", [e.name for e in selected])
                scores = compute_config_scores_config_mode(pairs, selected, noise_scale, args.otu_mode)
            else:
                scores = compute_config_scores_per_image_mode(
                    pairs, encoders, n_encoders, noise_scale, args.seed, args.num_runs
                )

            threshold, auc = choose_threshold(scores, labels, args.threshold_mode) if args.threshold == "auto" else (float(args.threshold), float("nan"))
            metrics = compute_metrics(scores, labels, threshold)
            metrics.update({
                "n_encoders": n_encoders,
                "otu_noise_scale": noise_scale,
                "threshold_mode": args.threshold_mode,
                "sampling_mode": args.sampling_mode,
                "otu_mode": args.otu_mode,
                "auc_clean_positive": auc,
                "score_clean_mean": float(scores[labels == 0].mean()),
                "score_adv_mean": float(scores[labels == 1].mean()),
                "score_gap": float(scores[labels == 0].mean() - scores[labels == 1].mean()),
            })
            summary_rows.append(metrics)
            print(json.dumps(metrics, indent=2))

            for item, score in zip(pairs, scores):
                all_score_rows.append({
                    "n_encoders": n_encoders,
                    "otu_noise_scale": noise_scale,
                    "input": str(item["input"]),
                    "generated": str(item["generated"]),
                    "label": int(item["label"]),
                    "label_name": "adversarial" if int(item["label"]) == 1 else "clean",
                    "score_mean": float(score),
                    "score_std": 0.0,
                    "num_runs": args.num_runs,
                })

    score_fields = ["n_encoders", "otu_noise_scale", "input", "generated", "label", "label_name", "score_mean", "score_std", "num_runs"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fields)
        writer.writeheader()
        writer.writerows(all_score_rows)

    summary_fields = [
        "n_encoders", "otu_noise_scale", "threshold_mode", "sampling_mode", "otu_mode",
        "threshold", "accuracy", "tpr", "fpr", "precision", "auc_clean_positive",
        "tp", "fp", "tn", "fn", "n_total", "n_clean", "n_adversarial",
        "score_clean_mean", "score_adv_mean", "score_gap",
    ]
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nWrote scores to: {args.out_csv}")
    print(f"Wrote summary to: {args.summary_csv}")
    print(f"Wrote summary JSON to: {args.summary_json}")


if __name__ == "__main__":
    main()
