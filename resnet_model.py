#!/usr/bin/env python
"""Train ResNet50-based IDS model on wavelet-transformed data."""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def build_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_dataset(data_path, label_path):
    data = np.load(data_path)
    data_resized_np = tf.image.resize(data, [224, 224]).numpy()

    labels_df = pd.read_csv(label_path)
    labels = labels_df["Label"].values
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    binary_labels = to_categorical(integer_encoded)
    return data_resized_np, binary_labels, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 model for TOW-IDS.")
    parser.add_argument(
        "--data-path",
        default="wavelet_transformed_data/layered_452x452_normalized.npy",
        help="Path to wavelet-transformed training data (.npy).",
    )
    parser.add_argument("--label-path", default="labels/452x452.csv", help="Path to labels CSV.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--output-model",
        default="models/452x452_trained_resnet_model.h5",
        help="Output model path.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU and run training on CPU.",
    )
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    x_data, y_data, label_encoder = load_dataset(args.data_path, args.label_path)
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.1, random_state=42
    )

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model = build_model()
    steps = max(1, len(x_train) // args.batch_size)
    model.fit(
        datagen.flow(x_train, y_train, shuffle=True, batch_size=args.batch_size),
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
    )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model, save_format="h5")
    print(f"Model saved to {args.output_model}")

    predictions = model.predict(x_val[:10], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    print("First 10 validation predictions:", predicted_labels.tolist())


if __name__ == "__main__":
    main()
