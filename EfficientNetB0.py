#!/usr/bin/env python
"""Train EfficientNetB0 IDS model on wavelet-transformed data."""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def build_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_dataset(data_path, label_path):
    wavelet_data = np.load(data_path)
    labels_df = pd.read_csv(label_path)
    labels = labels_df["Label"].values

    indices = np.arange(wavelet_data.shape[0])
    np.random.shuffle(indices)
    wavelet_data = wavelet_data[indices]
    labels = labels[indices]

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    binary_labels = to_categorical(integer_encoded)
    data_resized = tf.image.resize(wavelet_data, [224, 224]).numpy()
    return data_resized, binary_labels, label_encoder


def scheduler(epoch, lr):
    return lr if epoch < 5 else lr * 0.9


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 model for TOW-IDS.")
    parser.add_argument(
        "--data-path",
        default="wavelet_transformed_data/layered_60x60_normalized.npy",
        help="Path to wavelet-transformed training data (.npy).",
    )
    parser.add_argument("--label-path", default="labels/60x60.csv", help="Path to labels CSV.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument(
        "--output-model",
        default="models/60x60_trained_EfficientNet_model.h5",
        help="Output model path.",
    )
    args = parser.parse_args()

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
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(scheduler)
    steps = max(1, len(x_train) // args.batch_size)
    model.fit(
        datagen.flow(x_train, y_train, shuffle=True, batch_size=args.batch_size),
        steps_per_epoch=steps,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, lr_scheduler],
    )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model, save_format="h5")
    print(f"Model saved to {args.output_model}")

    predictions = model.predict(x_val[:10], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    print("First 10 validation predictions:", label_encoder.inverse_transform(predicted_classes).tolist())


if __name__ == "__main__":
    main()
