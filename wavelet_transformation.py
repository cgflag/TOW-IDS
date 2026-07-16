#!/usr/bin/env python
"""Apply wavelet transform and auto-save visualization figures."""

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavelet_features import transform_dataset

NORMALIZED_FOLDER = "normalized_packet_data"
WAVELET_TRANSFORMED_FOLDER = "wavelet_transformed_data"
TARGET_FILES = [
    "32x32_normalized.npy",
    "60x60_normalized.npy",
    "116x116_normalized.npy",
    "228x228_normalized.npy",
    "452x452_normalized.npy",
]
SUPPORTED_VARIANTS = ["three_wavelets_ll", "haar_four_subbands", "haar_ll"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_figure_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_dir = Path("figures") / f"train_wavelet_{timestamp}"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def save_channel_grid(images: np.ndarray, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wavelet-transformed training features.")
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="three_wavelets_ll",
        help="Wavelet feature variant.",
    )
    parser.add_argument("--level", type=int, default=1, help="Recursive DWT level (>=1).")
    args = parser.parse_args()

    ensure_dir(WAVELET_TRANSFORMED_FOLDER)
    figure_dir = create_figure_output_dir()
    print(f"Figure output directory: {figure_dir}")
    print(f"Wavelet variant: {args.variant}, level: {args.level}")

    for file_name in TARGET_FILES:
        normalized_file_path = os.path.join(NORMALIZED_FOLDER, file_name)
        image_data = np.load(normalized_file_path)
        layered_data_array = transform_dataset(image_data, variant=args.variant, level=args.level)
        variant_tag = f"{args.variant}_lv{args.level}"
        if args.variant == "three_wavelets_ll" and args.level == 1:
            save_file_name = os.path.join(WAVELET_TRANSFORMED_FOLDER, f"layered_{file_name}")
        else:
            save_file_name = os.path.join(
                WAVELET_TRANSFORMED_FOLDER,
                f"{variant_tag}_layered_{file_name}",
            )
        np.save(save_file_name, layered_data_array)
        print(f"Layered data saved to {save_file_name}")

        # Auto-save preview figures with semantic names.
        preview_count = min(5, len(layered_data_array))
        if preview_count == 0:
            continue
        preview_images = layered_data_array[:preview_count]
        size_tag = file_name.replace("_normalized.npy", "")
        for channel in range(layered_data_array.shape[-1]):
            channel_images = preview_images[:, :, :, channel]
            # Fill to 5 slots for consistent layout if dataset is tiny.
            if preview_count < 5:
                pad_img = np.zeros_like(channel_images[0])
                channel_images = np.concatenate(
                    [channel_images, np.stack([pad_img] * (5 - preview_count), axis=0)],
                    axis=0,
                )
            figure_name = (
                f"train_{variant_tag}_{size_tag}_channel{channel + 1}_preview.png"
            )
            save_channel_grid(
                channel_images,
                title=f"{variant_tag} {size_tag} channel {channel + 1}",
                save_path=figure_dir / figure_name,
            )
            print(f"Saved figure: {figure_dir / figure_name}")

    file_names = [f for f in os.listdir(WAVELET_TRANSFORMED_FOLDER) if f.endswith(".npy")]
    for file_name in file_names:
        file_path = os.path.join(WAVELET_TRANSFORMED_FOLDER, file_name)
        data = np.load(file_path)
        print(f"File: {file_name}, Shape: {data.shape}")
    print(f"Total number of files: {len(file_names)}")


if __name__ == "__main__":
    main()
