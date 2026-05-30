# MirrorCheck

**MirrorCheck** is a training-free, model-agnostic adversarial detection framework for vision-language models. It detects attacks by checking whether the model's textual interpretation can visually reconstruct an image that remains semantically consistent with the original input.

## Core idea

Given an input image `x_in`, a victim model produces text `t`. A text-to-image model generates `x_gen` from `t`. MirrorCheck compares the embeddings of `x_in` and `x_gen` using image encoders. Low similarity indicates a likely adversarial input.

```text
input image -> victim model text -> T2I generated image -> image-image similarity -> decision
```

## Installation

COMING SOON...
