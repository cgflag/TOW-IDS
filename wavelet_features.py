#!/usr/bin/env python
"""Wavelet feature builders for table2 experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pywt

WaveletVariant = Literal["three_wavelets_ll", "haar_four_subbands", "haar_ll"]


def _recursive_ll(image: np.ndarray, wavelet_name: str, level: int) -> np.ndarray:
    ll = image
    for _ in range(level):
        ll, _ = pywt.dwt2(ll, wavelet_name)
    return ll


def _recursive_final_subbands(image: np.ndarray, wavelet_name: str, level: int):
    base = image
    for _ in range(max(0, level - 1)):
        base, _ = pywt.dwt2(base, wavelet_name)
    ll, (lh, hl, hh) = pywt.dwt2(base, wavelet_name)
    return ll, lh, hl, hh


def transform_image(
    image: np.ndarray,
    *,
    variant: WaveletVariant,
    level: int,
) -> np.ndarray:
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")

    if variant == "three_wavelets_ll":
        ll_coif1 = _recursive_ll(image, "coif1", level)
        ll_db3 = _recursive_ll(image, "db3", level)
        ll_rbio13 = _recursive_ll(image, "rbio1.3", level)
        return np.stack((ll_coif1, ll_db3, ll_rbio13), axis=-1)

    if variant == "haar_four_subbands":
        ll, lh, hl, hh = _recursive_final_subbands(image, "haar", level)
        return np.stack((ll, lh, hl, hh), axis=-1)

    if variant == "haar_ll":
        ll_haar = _recursive_ll(image, "haar", level)
        return np.expand_dims(ll_haar, axis=-1)

    raise ValueError(f"Unsupported variant: {variant}")


def transform_dataset(
    images: np.ndarray,
    *,
    variant: WaveletVariant,
    level: int,
) -> np.ndarray:
    transformed = [transform_image(img, variant=variant, level=level) for img in images]
    return np.array(transformed, dtype=np.float32)
