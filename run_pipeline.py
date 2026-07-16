#!/usr/bin/env python
"""Run end-to-end TOW-IDS pipeline with pure Python scripts."""

import argparse
import subprocess
import sys
from pathlib import Path

SUPPORTED_SIZES = [32, 60, 116, 228, 452]
TABLE2_SIZE_LEVELS = {
    32: [1],
    60: [1, 2],
    116: [1, 2, 3],
    228: [1, 2, 3, 4],
    452: [1, 2, 3, 4, 5],
}
TABLE2_VARIANTS = [
    "three_wavelets_1_subband",
    "one_wavelet_4_subbands",
    "one_wavelet_1_subband",
]


def run_step(command):
    print(f"\n[RUN] {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {result.returncode}: {' '.join(command)}")


def resolve_sizes(size: int, all_sizes: bool):
    if all_sizes:
        return SUPPORTED_SIZES
    return [size]


def main():
    parser = argparse.ArgumentParser(description="Run TOW-IDS data pipeline and model training.")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_all", "eval", "table2"],
        default="train",
        help=(
            "train: preprocessing + wavelet + custom model; "
            "test: test preprocessing + test wavelet; "
            "train_all: train all model scripts; "
            "eval: evaluate trained model on test data; "
            "table2: run paper-style variant/level comparison experiments."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run sub scripts.",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=SUPPORTED_SIZES,
        default=452,
        help="Target N x N size for train/eval modes (default: 452).",
    )
    parser.add_argument(
        "--all-sizes",
        action="store_true",
        help="Run train/eval for all supported sizes (32/60/116/228/452).",
    )
    parser.add_argument(
        "--eval-save-json",
        default="",
        help="When mode=eval with single size, optional metrics JSON output path.",
    )
    parser.add_argument(
        "--eval-save-json-template",
        default="results/eval_{size}x{size}.json",
        help="When mode=eval, JSON template for output path. Supports {size}.",
    )
    parser.add_argument(
        "--eval-table2-grid",
        action="store_true",
        help=(
            "When mode=eval, evaluate all table2 combinations "
            "(variant + size + level) using saved table2 artifacts."
        ),
    )
    parser.add_argument(
        "--eval-table2-model-dir",
        default="models/table2",
        help="Base directory of table2 models for --eval-table2-grid.",
    )
    parser.add_argument(
        "--eval-table2-test-data-dir",
        default="test_wavelet_transformed_data/table2",
        help="Base directory of table2 test data for --eval-table2-grid.",
    )
    parser.add_argument(
        "--eval-table2-save-json-template",
        default="results/eval_table2/{variant}_{size}x{size}_lv{level}.json",
        help=(
            "When --eval-table2-grid is enabled, JSON output template. "
            "Supports {variant}, {size}, {level}."
        ),
    )
    parser.add_argument(
        "--table2-epochs",
        type=int,
        default=30,
        help="When mode=table2, epochs per experiment setting.",
    )
    parser.add_argument(
        "--table2-batch-size",
        type=int,
        default=32,
        help="When mode=table2, batch size per experiment setting.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = args.python
    sizes = resolve_sizes(args.size, args.all_sizes)

    if args.mode in {"train", "train_all"}:
        run_step([py, str(root / "preprocessing.py")])
        run_step([py, str(root / "wavelet_transformation.py")])
        for size in sizes:
            run_step([py, str(root / "tow_ids_model.py"), "--size", str(size)])

    if args.mode == "train_all":
        run_step([py, str(root / "tow_ids_no_wavelet.py")])
        run_step([py, str(root / "resnet_model.py")])
        run_step([py, str(root / "EfficientNetB0.py")])

    if args.mode == "test":
        run_step([py, str(root / "test_preprocessing.py")])
        run_step([py, str(root / "test_wavelet_transformed.py")])

    if args.mode == "eval":
        if args.eval_table2_grid:
            model_dir = Path(args.eval_table2_model_dir)
            test_data_dir = Path(args.eval_table2_test_data_dir)
            target_sizes = set(sizes)
            total = sum(len(TABLE2_SIZE_LEVELS[size]) for size in target_sizes) * len(TABLE2_VARIANTS)
            current = 0
            for size in sorted(target_sizes):
                for level in TABLE2_SIZE_LEVELS[size]:
                    for variant in TABLE2_VARIANTS:
                        current += 1
                        print(
                            f"[EVAL {current}/{total}] variant={variant}, size={size}x{size}, level=Lv{level}"
                        )
                        model_path = model_dir / f"{variant}_{size}x{size}_lv{level}_tow-ids_model.h5"
                        test_data_path = test_data_dir / f"layered_test_{variant}_{size}x{size}_lv{level}.npy"
                        save_json = args.eval_table2_save_json_template.format(
                            variant=variant,
                            size=size,
                            level=level,
                        )
                        run_step(
                            [
                                py,
                                str(root / "evaluate.py"),
                                "--size",
                                str(size),
                                "--model-path",
                                str(model_path),
                                "--test-data-path",
                                str(test_data_path),
                                "--save-json",
                                save_json,
                            ]
                        )
        else:
            for size in sizes:
                if args.eval_save_json and len(sizes) == 1:
                    save_json = args.eval_save_json
                else:
                    save_json = args.eval_save_json_template.format(size=size)
                run_step(
                    [
                        py,
                        str(root / "evaluate.py"),
                        "--size",
                        str(size),
                        "--save-json",
                        save_json,
                    ]
                )

    if args.mode == "table2":
        run_step(
            [
                py,
                str(root / "table2_experiment.py"),
                "--epochs",
                str(args.table2_epochs),
                "--batch-size",
                str(args.table2_batch_size),
            ]
        )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
