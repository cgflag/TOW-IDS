#!/usr/bin/env python
"""Train custom TOW-IDS CNN model on wavelet-transformed data."""

import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    SeparableConv2D,
)
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

SUPPORTED_SIZES = [32, 60, 116, 228, 452]


def conv_block_a(x, filters):
    # 根据图7，Block A 包含两个 SeparableConv2D, BN 和 MaxPool
    x = layers.SeparableConv2D(filters, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(filters, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def conv_block_b(x, filters):
    # --- 残差连接的关键：保存输入 ---
    shortcut = x
    
    # Block B 的内部操作
    x = layers.SeparableConv2D(filters, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = layers.SeparableConv2D(filters, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # --- 执行直接残差加法：F(x) + x ---
    x = layers.Add()([x, shortcut])
    return x

def conv_block_c(x):
    # Block C 结构
    x = layers.SeparableConv2D(512, kernel_size=3, strides=1, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation="sigmoid")(x)
    return x

def build_model(input_shape):
    # 使用函数式 API 定义输入
    inputs = layers.Input(shape=input_shape)
    
    # 按照图6/图7流程连接各块：输入后直接进入 Block A
    x = conv_block_a(inputs, 256)
    
    # 3. Block B 循环 5 次
    for _ in range(5):
        x = conv_block_b(x, 256)
        
    # 4. Block C
    outputs = conv_block_c(x)
    
    # 封装模型
    model = models.Model(inputs=inputs, outputs=outputs, name="TOW_IDS_Functional")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_dataset(data_path, label_path):
    x_data = np.load(data_path)
    labels_df = pd.read_csv(label_path)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels_df["Label"].values)
    y_data = to_categorical(integer_encoded)
    return x_data, y_data, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train TOW-IDS model with wavelet features.")
    parser.add_argument(
        "--size",
        type=int,
        choices=SUPPORTED_SIZES,
        default=452,
        help="Target N x N size to train (default: 452).",
    )
    parser.add_argument(
        "--data-path",
        default="",
        help="Path to wavelet-transformed training data (.npy). Defaults to --size path.",
    )
    parser.add_argument(
        "--label-path",
        default="",
        help="Path to labels CSV. Defaults to --size path.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output-model",
        default="",
        help="Output model path. Defaults to --size path.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_path = args.data_path or f"wavelet_transformed_data/layered_{args.size}x{args.size}_normalized.npy"
    label_path = args.label_path or f"labels/{args.size}x{args.size}.csv"
    output_model = args.output_model or f"models/{args.size}x{args.size}_tow-ids_model.h5"

    x_data, y_data, label_encoder = load_dataset(data_path, label_path)
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, test_size=0.2, random_state=args.seed
    )

    model = build_model(input_shape=x_data.shape[1:])
    model.summary()
    model.fit(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
    )

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    model.save(output_model, save_format="h5")
    print(f"Model saved to {output_model}")

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
