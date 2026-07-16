#!/usr/bin/env python
"""Train custom TOW-IDS CNN model directly on normalized packet matrices."""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical


def conv_block_a(filters, input_shape=None):
    return Sequential(
        [
            SeparableConv2D(
                filters,
                kernel_size=3,
                strides=3,
                activation="relu",
                padding="same",
                input_shape=input_shape,
                kernel_regularizer=l2(0.001),
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
        ]
    )


def conv_block_b(filters):
    return Sequential(
        [
            SeparableConv2D(filters, kernel_size=3, strides=3, activation="relu", padding="same"),
            BatchNormalization(),
            Dropout(0.3),
        ]
    )


def conv_block_c():
    return Sequential(
        [
            SeparableConv2D(512, kernel_size=3, strides=3, activation="relu", padding="same"),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="sigmoid"),
        ]
    )


def build_model(input_shape):
    model = Sequential()
    model.add(conv_block_a(256, input_shape=input_shape))
    for _ in range(5):
        model.add(conv_block_b(256))
    model.add(conv_block_c())
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_dataset(data_path, label_path):
    x_data = np.load(data_path)
    x_data = np.expand_dims(x_data, axis=-1)
    labels_df = pd.read_csv(label_path)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels_df["Label"].values)
    y_data = to_categorical(integer_encoded)
    return x_data, y_data, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train TOW-IDS model without wavelet transform.")
    parser.add_argument(
        "--data-path",
        default="normalized_packet_data/452x452_normalized.npy",
        help="Path to normalized training data (.npy).",
    )
    parser.add_argument("--label-path", default="labels/452x452.csv", help="Path to labels CSV.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--output-model",
        default="models/452x452_tow-ids_model_no_wavelet.h5",
        help="Output model path.",
    )
    args = parser.parse_args()

    x_data, y_data, label_encoder = load_dataset(args.data_path, args.label_path)
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    model = build_model(input_shape=x_data.shape[1:])
    model.summary()
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model, save_format="h5")
    print(f"Model saved to {args.output_model}")

    sample_prediction = model.predict(np.expand_dims(x_val[0], axis=0), verbose=0)
    predicted_class = np.argmax(sample_prediction, axis=1)[0]
    true_class = np.argmax(y_val[0])
    print(
        "Sample validation prediction:",
        f"pred={label_encoder.inverse_transform([predicted_class])[0]},",
        f"true={label_encoder.inverse_transform([true_class])[0]}",
    )


if __name__ == "__main__":
    main()
