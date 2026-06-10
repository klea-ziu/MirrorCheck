# MirrorCheck

**MirrorCheck** is a training-free, model-agnostic adversarial detection framework for vision-language models. It detects attacks by checking whether the model's textual interpretation can visually reconstruct an image that remains semantically consistent with the original input.

## Links

- Paper page: https://www.norange.io/projects/mirrorcheck/
- Video: https://www.youtube.com/watch?v=3OEuU5bfNZQ

## Core idea

Given an input image `x_in`, a victim model produces text `t`. A text-to-image model generates `x_gen` from `t`. MirrorCheck compares the embeddings of `x_in` and `x_gen` using image encoders. Low similarity indicates a likely adversarial input.

```text
input image -> victim model text -> T2I generated image -> image-image similarity -> decision
```

## Installation

```bash
git clone https://github.com/klea-ziu/MirrorCheck.git
cd MirrorCheck
pip install -r requirements.txt
```

## Quick start

If you already have a generated image from the victim model output, run:

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

## Evaluating folders of image pairs

If you already have original images and T2I-generated images, use:

```bash
scripts/evaluate_stochastic_folder.py
```

The script supports three pairing modes.

### Name-based pairing

Use this when the original and generated images share the same filename.

```bash
--pairing name
```

Example:

```text
clean_img/example_001.png
generated_clean/example_001.png
```

### Index-based pairing

Use this when the original and generated images do **not** share the same filename, but their sorted orders match.

This is useful for folders like:

```text
Original images:  ILSVRC2012_val_00040001.png, ILSVRC2012_val_00040002.png, ...
Generated images: 00000.png, 00001.png, ...
```

Run with:

```bash
--pairing index
```

### CSV-based pairing

For the most explicit and reproducible pairing, provide a CSV mapping:

```csv
input,generated
ILSVRC2012_val_00040001.png,00000.png
ILSVRC2012_val_00040002.png,00001.png
```

Then run with:

```bash
--pairing csv
--mapping-csv /path/to/pairs.csv
```

## Stochastic MirrorCheck

The stochastic version randomizes the detection pipeline at inference time:

1. random text-to-image generator selection,
2. random image encoder subset selection,
3. one-time-use perturbation of selected encoder parameters.

The summary CSV reports the threshold, accuracy, TPR, FPR, precision, confusion counts, and mean clean/adversarial similarity scores for each stochastic configuration.


The original experiment used a 10-encoder zoo: 5 OpenAI CLIP encoders plus 5 OpenCLIP encoders. It sampled the encoder subset, applied OTU noise, and chose the ROC threshold.

To reproduce, use:

```bash
python scripts/evaluate_stochastic_folder.py \
  --clean-dir /path/to/clean_img \
  --clean-generated-dir /path/to/generated_clean/samples \
  --adv-dir /path/to/adv_img \
  --adv-generated-dir /path/to/generated_adv/samples \
  --pairing index \
  --encoder-preset legacy_10 \
  --n-encoders 1,3,5,7,10 \
  --otu-noise-scale 5e-6 \
  --max-samples 100 \
  --sampling-mode config \
  --otu-mode legacy_once \
  --threshold-mode equal_tpr_tnr \
  --seed 42 \
  --out-csv stochastic_scores.csv \
  --summary-csv stochastic_summary.csv
```


### Full stochastic grid

To evaluate multiple encoder counts and OTU noise scales, run:

```bash
python scripts/evaluate_stochastic_folder.py \
  --clean-dir /path/to/clean_img \
  --clean-generated-dir /path/to/generated_clean/samples \
  --adv-dir /path/to/adv_img \
  --adv-generated-dir /path/to/generated_adv/samples \
  --pairing index \
  --encoder-preset legacy_10 \
  --n-encoders 1,3,5,7,10 \
  --otu-noise-scale 5e-6,5e-4,1e-3 \
  --sampling-mode config \
  --otu-mode legacy_once \
  --threshold-mode equal_tpr_tnr \
  --seed 42 \
  --out-csv stochastic_scores_grid.csv \
  --summary-csv stochastic_summary_grid.csv
```

### Custom encoder list

You can also provide a custom encoder list:

```bash
python scripts/evaluate_stochastic_folder.py \
  --clean-dir /path/to/clean_img \
  --clean-generated-dir /path/to/generated_clean/samples \
  --adv-dir /path/to/adv_img \
  --adv-generated-dir /path/to/generated_adv/samples \
  --pairing index \
  --encoders ViT-B-32:open_clip:laion400m_e31,RN50:open_clip:yfcc15m,ViT-L-14:open_clip:laion400m_e31 \
  --n-encoders 1,3 \
  --otu-noise-scale 5e-6 \
  --threshold-mode equal_tpr_tnr \
  --out-csv stochastic_scores.csv \
  --summary-csv stochastic_summary.csv
```

Encoder specifications follow the format:

```text
model_name:library:pretrained
```

For example:

```text
RN50:clip
ViT-B/32:clip
RN50:open_clip:yfcc15m
ViT-B-32:open_clip:laion400m_e31
```

## Attacks

For generating adversarial examples, we used the original implementations of the corresponding attack methods. This repository focuses on the MirrorCheck detection pipeline.

- AttackVLM: https://github.com/yunqing-me/attackvlm
- VLAttack: https://github.com/ericyinyzy/VLAttack
- Attack-Bard: https://github.com/thu-ml/Attack-Bard

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
