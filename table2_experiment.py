#!/usr/bin/env python
"""Run Table-II style wavelet/level comparison experiments."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Prefer dedicated GPU only for DirectML stability.
os.environ.setdefault("DML_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from tow_ids_model import build_model
from wavelet_features import transform_dataset

SIZES = [32, 60, 116, 228, 452]
SIZE_LEVELS = {
    32: [1],
    60: [1, 2],
    116: [1, 2, 3],
    228: [1, 2, 3, 4],
    452: [1, 2, 3, 4, 5],
}

VARIANT_MAP = {
    "three_wavelets_1_subband": "three_wavelets_ll",
    "one_wavelet_4_subbands": "haar_four_subbands",
    "one_wavelet_1_subband": "haar_ll",
}


@dataclass
class ExperimentResult:
    wavelet_type: str
    size: int
    level: int
    accuracy: float
    f_measure_macro: float
    f_measure_abnormal: float
    model_path: str
    test_data_path: str
    eval_json_path: str


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_labels(label_path: Path) -> List[str]:
    labels_df = pd.read_csv(label_path)
    if "Label" not in labels_df.columns:
        raise ValueError(f"Label column missing in {label_path}")
    return labels_df["Label"].astype(str).tolist()


def run_single_experiment(
    *,
    size: int,
    level: int,
    variant_key: str,
    epochs: int,
    batch_size: int,
    seed: int,
    cache_dir: Path,
    model_dir: Path,
    test_data_dir: Path,
    eval_dir: Path,
    save_models: bool,
    save_eval_json: bool,
    skip_existing: bool,
    safe_batch_for_large_inputs: bool,
) -> ExperimentResult:
    tf.keras.backend.clear_session()
    gc.collect()
    variant = VARIANT_MAP[variant_key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_path = cache_dir / f"train_{variant_key}_{size}x{size}_lv{level}.npy"
    test_cache_path = cache_dir / f"test_{variant_key}_{size}x{size}_lv{level}.npy"

    test_data_dir.mkdir(parents=True, exist_ok=True)
    test_data_path = test_data_dir / f"layered_test_{variant_key}_{size}x{size}_lv{level}.npy"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{variant_key}_{size}x{size}_lv{level}_tow-ids_model.h5"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = eval_dir / f"{variant_key}_{size}x{size}_lv{level}.json"

    if skip_existing and model_path.exists() and test_data_path.exists() and eval_json_path.exists():
        with eval_json_path.open("r", encoding="utf-8") as f:
            cached_eval = json.load(f)
        print(f"  -> skip training, reuse existing artifacts: {model_path.name}")
        return ExperimentResult(
            wavelet_type=variant_key,
            size=size,
            level=level,
            accuracy=float(cached_eval["accuracy"]),
            f_measure_macro=float(cached_eval.get("f_measure_macro", cached_eval["f1_score"])),
            f_measure_abnormal=float(cached_eval.get("f_measure_abnormal", cached_eval["f1_score"])),
            model_path=str(model_path),
            test_data_path=str(test_data_path),
            eval_json_path=str(eval_json_path),
        )

    test_labels_text = load_labels(Path("test_labels") / f"{size}x{size}.csv")
    test_data = np.load(Path("test_normalized_packet_data") / f"test_{size}x{size}_normalized.npy")
    if len(test_data) != len(test_labels_text):
        raise ValueError(f"Test data/label mismatch for {size}: {len(test_data)} vs {len(test_labels_text)}")

    if test_data_path.exists():
        x_test = np.load(test_data_path)
    elif test_cache_path.exists():
        x_test = np.load(test_cache_path)
        np.save(test_data_path, x_test)
    else:
        x_test = transform_dataset(test_data, variant=variant, level=level)
        np.save(test_cache_path, x_test)
        np.save(test_data_path, x_test)

    encoder = LabelEncoder()
    train_labels_text = load_labels(Path("labels") / f"{size}x{size}.csv")
    y_train_int = encoder.fit_transform(train_labels_text)
    y_test_int = encoder.transform(test_labels_text)

    set_global_seed(seed)
    if skip_existing and model_path.exists():
        print(f"  -> skip training, load existing model: {model_path.name}")
        model = load_model(model_path)
    else:
        train_data = np.load(Path("normalized_packet_data") / f"{size}x{size}_normalized.npy")
        if len(train_data) != len(train_labels_text):
            raise ValueError(
                f"Train data/label mismatch for {size}: {len(train_data)} vs {len(train_labels_text)}"
            )

        if train_cache_path.exists():
            x_train = np.load(train_cache_path)
        else:
            x_train = transform_dataset(train_data, variant=variant, level=level)
            np.save(train_cache_path, x_train)
        y_train = to_categorical(y_train_int)

        effective_batch_size = batch_size
        if safe_batch_for_large_inputs:
            if size >= 452:
                effective_batch_size = min(batch_size, 2)
            elif size >= 228:
                effective_batch_size = min(batch_size, 8)
            elif size >= 116:
                effective_batch_size = min(batch_size, 16)

        model = build_model(input_shape=x_train.shape[1:])
        callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
        model.fit(
            x_train,
            y_train,
            batch_size=effective_batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )
        if save_models:
            model.save(model_path, save_format="h5")

    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    abnormal_idx = int(np.where(encoder.classes_ == "Abnormal")[0][0])
    precision, recall, f1_binary, _ = precision_recall_fscore_support(
        y_test_int,
        y_pred,
        average="binary",
        pos_label=abnormal_idx,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_int, y_pred)
    eval_payload = {
        "wavelet_type": variant_key,
        "image_size": f"{size}x{size}",
        "level": level,
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
        "test_label_path": str(Path("test_labels") / f"{size}x{size}.csv"),
        "classes": list(encoder.classes_),
        "accuracy": float(accuracy_score(y_test_int, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_binary),
        "f_measure_macro": float(f1_score(y_test_int, y_pred, average="macro", zero_division=0)),
        "f_measure_abnormal": float(
            f1_score(y_test_int, y_pred, labels=[abnormal_idx], average="macro", zero_division=0)
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test_int,
            y_pred,
            target_names=list(encoder.classes_),
            output_dict=True,
            zero_division=0,
        ),
    }
    if save_eval_json:
        with eval_json_path.open("w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2, ensure_ascii=False)

    # Release graph/resources before next table setting.
    tf.keras.backend.clear_session()
    gc.collect()

    return ExperimentResult(
        wavelet_type=variant_key,
        size=size,
        level=level,
        accuracy=eval_payload["accuracy"],
        f_measure_macro=eval_payload["f_measure_macro"],
        f_measure_abnormal=eval_payload["f_measure_abnormal"],
        model_path=str(model_path),
        test_data_path=str(test_data_path),
        eval_json_path=str(eval_json_path),
    )


def write_results(results: List[ExperimentResult], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "wavelet_type",
                "image_size",
                "level",
                "accuracy",
                "f_measure_macro",
                "f_measure_abnormal",
                "model_path",
                "test_data_path",
                "eval_json_path",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.wavelet_type,
                    f"{item.size}x{item.size}",
                    f"Lv{item.level}",
                    f"{item.accuracy:.4f}",
                    f"{item.f_measure_macro:.4f}",
                    f"{item.f_measure_abnormal:.4f}",
                    item.model_path,
                    item.test_data_path,
                    item.eval_json_path,
                ]
            )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "wavelet_type": item.wavelet_type,
            "image_size": f"{item.size}x{item.size}",
            "level": item.level,
            "accuracy": item.accuracy,
            "f_measure_macro": item.f_measure_macro,
            "f_measure_abnormal": item.f_measure_abnormal,
            "model_path": item.model_path,
            "test_data_path": item.test_data_path,
            "eval_json_path": item.eval_json_path,
        }
        for item in results
    ]
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table-II style comparison experiments.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per setting.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--out-csv",
        default="results/table2_results.csv",
        help="Output CSV path for comparison table.",
    )
    parser.add_argument(
        "--out-json",
        default="results/table2_results.json",
        help="Output JSON path for raw records.",
    )
    parser.add_argument(
        "--cache-dir",
        default="wavelet_experiments/cache",
        help="Directory for transformed feature cache files.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/table2",
        help="Directory to save per-setting trained models.",
    )
    parser.add_argument(
        "--test-data-dir",
        default="test_wavelet_transformed_data/table2",
        help="Directory to save per-setting transformed test features for eval.",
    )
    parser.add_argument(
        "--eval-dir",
        default="results/table2_eval",
        help="Directory to save per-setting eval JSON files.",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save a trained model file for each experiment setting.",
    )
    parser.add_argument(
        "--save-eval-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save a detailed eval JSON for each experiment setting.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip training when model/test/eval artifacts already exist for a setting.",
    )
    parser.add_argument(
        "--safe-batch-for-large-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically reduce batch size for 116/228/452 inputs to avoid GPU timeout.",
    )
    args = parser.parse_args()

    records: List[ExperimentResult] = []
    total = sum(len(SIZE_LEVELS[size]) for size in SIZES) * len(VARIANT_MAP)
    current = 0

    for size in SIZES:
        for level in SIZE_LEVELS[size]:
            for variant_key in VARIANT_MAP:
                current += 1
                print(
                    f"[{current}/{total}] Running variant={variant_key}, size={size}x{size}, level=Lv{level}"
                )
                record = run_single_experiment(
                    size=size,
                    level=level,
                    variant_key=variant_key,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    cache_dir=Path(args.cache_dir),
                    model_dir=Path(args.model_dir),
                    test_data_dir=Path(args.test_data_dir),
                    eval_dir=Path(args.eval_dir),
                    save_models=args.save_models,
                    save_eval_json=args.save_eval_json,
                    skip_existing=args.skip_existing,
                    safe_batch_for_large_inputs=args.safe_batch_for_large_inputs,
                )
                records.append(record)
                print(
                    f"  -> accuracy={record.accuracy:.4f}, "
                    f"f_macro={record.f_measure_macro:.4f}, "
                    f"f_abnormal={record.f_measure_abnormal:.4f}"
                )

    write_results(records, Path(args.out_csv), Path(args.out_json))
    print(f"Saved table CSV: {args.out_csv}")
    print(f"Saved table JSON: {args.out_json}")


if __name__ == "__main__":
    main()
