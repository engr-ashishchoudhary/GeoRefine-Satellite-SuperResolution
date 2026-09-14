"""
Test-time augmentation (TTA) transforms for uncertainty estimation.

Why TTA rather than Monte Carlo Dropout: the pretrained RRDBNet
architecture (src/model/rrdbnet.py) has no dropout layers, and adding them
would require retraining - out of scope per the project's model-strategy
rules (prefer pretrained, do not over-invest in architecture changes).
TTA is a standard, free, no-retraining alternative: run inference on
several geometrically-equivalent views of the same input, undo the
transform on each output, and measure how much the resulting predictions
disagree. Disagreement across views that are mathematically equivalent
inputs is a legitimate (if approximate) signal of reconstruction
instability - see src/uncertainty/README.md and the project's top-level
scientific honesty notice for the cautious framing this is presented with.

Only horizontal/vertical flips are used (not 90-degree rotations), because
rotating by 90 degrees swaps height and width, which would complicate the
tiling math in src.inference.scene_inference.run_tiled_inference() for
non-square scenes without adding much beyond what flips already capture.

All four transforms here are involutions (applying the same transform
twice returns the original array), so invert_augmentation() simply calls
apply_augmentation() again with the same name - documented explicitly
rather than left implicit, since it would not generalize to a future
augmentation (e.g. a 90-degree rotation) added without updating this
assumption.
"""
from __future__ import annotations

from typing import List

import numpy as np

AUGMENTATION_NAMES = ("identity", "hflip", "vflip", "hvflip")


def apply_augmentation(array: np.ndarray, name: str) -> np.ndarray:
    """Apply a named augmentation to a (bands, H, W) array.

    "hflip" flips along the width axis (axis=2), "vflip" along the height
    axis (axis=1), "hvflip" both. Returns a new array (flips are views in
    numpy; callers should not rely on this being a copy vs. a view).
    """
    if name == "identity":
        return array
    if name == "hflip":
        return np.flip(array, axis=2)
    if name == "vflip":
        return np.flip(array, axis=1)
    if name == "hvflip":
        return np.flip(array, axis=(1, 2))
    raise ValueError(f"Unknown augmentation '{name}'. Expected one of {AUGMENTATION_NAMES}")


def invert_augmentation(array: np.ndarray, name: str) -> np.ndarray:
    """Undo a named augmentation on a (bands, H, W) array.

    Every augmentation defined here is its own inverse (flipping twice
    returns the original), so this is identical to apply_augmentation().
    Kept as a separate named function for readability at call sites and so
    a future non-involutory augmentation doesn't silently break this
    assumption without a visible place to fix it.
    """
    return apply_augmentation(array, name)


def validate_augmentation_names(names: List[str]) -> None:
    """Raise ValueError listing all unrecognized names, if any."""
    unknown = [n for n in names if n not in AUGMENTATION_NAMES]
    if unknown:
        raise ValueError(f"Unknown augmentation name(s) {unknown}. Expected a subset of {AUGMENTATION_NAMES}")
