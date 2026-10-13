"""Arquitecturas CNN: simple (desde cero) + transfer learning MobileNetV2."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import IMG_SIZE, LEARNING_RATE, NUM_CLASSES


def build_simple_cnn(input_shape=(128, 128, 3), num_classes=NUM_CLASSES, lr=1e-3):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="simple_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_mobilenetv2(
    input_shape=(128, 128, 3),
    num_classes=NUM_CLASSES,
    lr=LEARNING_RATE,
    dropout=0.3,
    freeze_base=True,
    weights="imagenet",
):
    base = keras.applications.MobileNetV2(
        include_top=False, input_shape=input_shape, weights=weights, pooling="avg"
    )
    base.trainable = not freeze_base
    inputs = keras.Input(shape=input_shape)
    # MobileNetV2 espera [0,1] o [-1,1]; como normalizamos a [0,1], reescalamos a [-1,1]
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base(x, training=False if freeze_base else True)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2_ethnicity")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(kind="transfer", **kwargs):
    size = kwargs.pop("img_size", IMG_SIZE)
    shape = (size, size, 3)
    if kind == "simple":
        return build_simple_cnn(input_shape=shape, **kwargs)
    return build_transfer_mobilenetv2(input_shape=shape, **kwargs)
