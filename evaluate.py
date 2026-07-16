#!/usr/bin/env python
"""Evaluate trained TOW-IDS models on test data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

SUPPORTED_SIZES = [32, 60, 116, 228, 452]


def read_label_csv(label_path: str) -> np.ndarray:
    labels_df = pd.read_csv(label_path)
    if "Label" not in labels_df.columns:
        raise ValueError(f"'Label' column not found in {label_path}")
    return labels_df["Label"].astype(str).values


def fit_label_encoder(train_label_path: str) -> LabelEncoder:
    train_labels = read_label_csv(train_label_path)
    encoder = LabelEncoder()
    encoder.fit(train_labels)
    return encoder


def align_input_shape(x_data: np.ndarray, model_input_shape) -> np.ndarray:
    """
    Make test data shape compatible with model input shape.
    Handles common cases:
    - (N, H, W) -> (N, H, W, 1)
    - (N, H, W, 1) -> (N, H, W) when model expects 3D
    """
    expected_rank = len(model_input_shape)
    current_rank = len(x_data.shape)

    if expected_rank == current_rank:
        return x_data

    if expected_rank == 4 and current_rank == 3:
        return np.expand_dims(x_data, axis=-1)

    if expected_rank == 3 and current_rank == 4 and x_data.shape[-1] == 1:
        return np.squeeze(x_data, axis=-1)

    raise ValueError(
        f"Input rank mismatch: model expects {model_input_shape}, "
        f"but data shape is {x_data.shape}"
    )


def predict_classes(model, x_data: np.ndarray) -> np.ndarray:
    y_pred_prob = model.predict(x_data, verbose=0)

    # Binary output in a single neuron
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
        return (y_pred_prob[:, 0] >= 0.5).astype(int)

    # Multi-class / 2-class with two logits/probs
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] >= 2:
        return np.argmax(y_pred_prob, axis=1)

    raise ValueError(f"Unsupported prediction shape: {y_pred_prob.shape}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TOW-IDS model performance.")
    parser.add_argument(
        "--size",
        type=int,
        choices=SUPPORTED_SIZES,
        default=452,
        help="Target N x N size to evaluate (default: 452).",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Path to trained model file. Defaults to --size path.",
    )
    parser.add_argument(
        "--test-data-path",
        default="",
        help="Path to test data (.npy). Defaults to --size path.",
    )
    parser.add_argument(
        "--test-label-path",
        default="",
        help="Path to test label CSV. Defaults to --size path.",
    )
    parser.add_argument(
        "--train-label-path",
        default="",
        help="Path to training label CSV used to fit LabelEncoder. Defaults to --size path.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional output path for metrics JSON.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path or f"models/{args.size}x{args.size}_tow-ids_model.h5")
    test_data_path = Path(
        args.test_data_path
        or f"test_wavelet_transformed_data/layered_test_{args.size}x{args.size}_normalized.npy"
    )
    test_label_path = Path(args.test_label_path or f"test_labels/{args.size}x{args.size}.csv")
    train_label_path = Path(args.train_label_path or f"labels/{args.size}x{args.size}.csv")

    model = load_model(model_path)
    x_test = np.load(test_data_path)
    x_test = align_input_shape(x_test, model.input_shape)

    label_encoder = fit_label_encoder(str(train_label_path))
    y_true_text = read_label_csv(str(test_label_path))
    y_true = label_encoder.transform(y_true_text)

    if len(x_test) != len(y_true):
        raise ValueError(
            f"Sample count mismatch: X has {len(x_test)} samples, "
            f"but y has {len(y_true)} labels."
        )

    y_pred = predict_classes(model, x_test)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    class_names = list(label_encoder.classes_)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path} | shape={x_test.shape}")
    print(f"Test labels: {test_label_path} | samples={len(y_true)}")
    print(f"Label classes: {class_names}")
    print()
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1-score : {f1:.6f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=6, zero_division=0))

    if args.save_json:
        result = {
            "model_path": str(model_path),
            "test_data_path": str(test_data_path),
            "test_label_path": str(test_label_path),
            "classes": class_names,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
        }
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics JSON to: {output_path}")


if __name__ == "__main__":
    main()
