# MirrorCheck

**MirrorCheck** is a training-free, model-agnostic adversarial detection framework for vision-language models. It detects attacks by checking whether the model's textual interpretation can visually reconstruct an image that remains semantically consistent with the original input.

## Core idea

Given an input image `x_in`, a victim model produces text `t`. A text-to-image model generates `x_gen` from `t`. MirrorCheck compares the embeddings of `x_in` and `x_gen` using image encoders. Low similarity indicates a likely adversarial input.

```text
input image -> victim model text -> T2I generated image -> image-image similarity -> decision
```

## Installation

```bash
git clone https://github.com/<your-org>/MirrorCheck.git
cd MirrorCheck
pip install -r requirements.txt
```

## Quick start

If you already have a generated image from the victim model output:

```bash
python scripts/run_detection.py \
  --image examples/clean/example.png \
  --text "a car parked on the street" \
  --generated-image examples/generated/example.png \
  --threshold 0.55
```

To run text-to-image generation directly, omit `--generated-image`:

```bash
python scripts/run_detection.py \
  --image examples/clean/example.png \
  --text "a car parked on the street" \
  --threshold 0.55
```

## Evaluate a folder of pairs

The filenames in `--input-dir` and `--generated-dir` should match.

```bash
python scripts/evaluate_folder.py \
  --input-dir examples/clean \
  --generated-dir examples/generated \
  --threshold 0.55 \
  --out-csv scores.csv
```

## Stochastic MirrorCheck

The stochastic version randomizes the detection pipeline at inference time:

1. random T2I generator selection,
2. random image encoder subset selection,
3. one-time-use perturbation of selected encoder parameters.

This makes the detector a moving target for adaptive attackers.

## Attacks

For generating adversarial examples, we used the original implementations of the corresponding attack methods. This repository focuses on the MirrorCheck detection pipeline.

- AttackVLM: https://github.com/yunqing-me/attackvlm
- VLAttack: https://github.com/ericyinyzy/VLAttack
- Attack-Bard: https://github.com/thu-ml/Attack-Bard

## Repository structure

```text
MirrorCheck/
  README.md
  requirements.txt
  mirrorcheck/
    __init__.py
    t2i.py
    encoders.py
    similarity.py
    detector.py
    stochastic.py
  examples/
    clean/
    adversarial/
    generated/
  scripts/
    run_detection.py
    evaluate_folder.py
  notebooks/
    mirrorcheck_demo.ipynb
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{fares2026mirrorcheck,
  title={MirrorCheck: Efficient Adversarial Defense for Vision-Language Models},
  author={Fares, Samar and Ziu, Klea and Aremu, Toluwani and Durasov, Nikita and Taka\v{c}, Martin and Fua, Pascal and Laptev, Ivan and Nandakumar, Karthik},
  booktitle={CVPR Workshops},
  year={2026}
}
```
