"""tf.data-Pipeline mit Augmentation zur Simulation von Smartphone-Bedingungen.

Liest die Manifest-CSVs (filepath,label) und baut performante tf.data-Datasets.
Die Trainings-Augmentation ahmt gezielt Alltagsfotos nach: zufällige Rotation,
Zoom/Crop, Farb-/Helligkeits-/Kontrastverschiebungen und leichte Gaußsche Unschärfe.
So wird das Modell robuster gegen Kameravariabilität als bei sauberen Klinikbildern.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras-2-Verhalten unter TF>=2.16

from typing import Any

import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _decode_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    # decode_image beherrscht PNG (PAD-UFES-20) UND JPG; expand_animations=False
    # liefert einen 3D-Tensor (H, W, C), damit resize funktioniert.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (img_size, img_size))
    return img  # float32 in [0, 255]; Normalisierung macht das EfficientNet-Preprocessing


def _gaussian_blur(img: tf.Tensor, sigma: tf.Tensor, kernel_size: int = 5) -> tf.Tensor:
    """Wendet depthwise einen Gaußschen Weichzeichner an (leichte Unschärfe)."""
    ax = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, 3, 1])  # gleicher Kernel pro Kanal
    img4 = img[tf.newaxis, ...]
    blurred = tf.nn.depthwise_conv2d(img4, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(blurred, axis=0)


def _augment(img: tf.Tensor, aug: dict[str, Any], img_size: int) -> tf.Tensor:
    # Rotation (über zufälliges 90°-Vielfaches + kleine affine Näherung via Zoom/Crop)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Zufälliger Zoom/Crop -> simuliert unterschiedlichen Abstand zur Läsion
    zoom = aug["zoom"]
    scale = tf.random.uniform([], 1.0 - zoom, 1.0 + zoom)
    new_size = tf.cast(tf.cast(img_size, tf.float32) * scale, tf.int32)
    img = tf.image.resize(img, (new_size, new_size))
    img = tf.image.resize_with_crop_or_pad(img, img_size, img_size)

    # Farb-/Helligkeits-Jitter -> unterschiedliche Kameras/Beleuchtung
    img = tf.image.random_brightness(img, aug["brightness"] * 255.0)
    img = tf.image.random_contrast(img, 1 - aug["contrast"], 1 + aug["contrast"])
    img = tf.image.random_hue(img, aug["hue"])
    img = tf.image.random_saturation(img, 1 - aug["saturation"], 1 + aug["saturation"])

    # Leichte Unschärfe mit Wahrscheinlichkeit blur_prob -> Autofokus-/Bewegungsunschärfe
    def do_blur():
        sigma = tf.random.uniform([], 0.3, aug["blur_max_sigma"])
        return _gaussian_blur(img, sigma)
    img = tf.cond(tf.random.uniform([]) < aug["blur_prob"], do_blur, lambda: img)

    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def build_dataset(csv_path: str, cfg: dict[str, Any], training: bool) -> tf.data.Dataset:
    df = pd.read_csv(csv_path)
    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].astype("float32").tolist()

    img_size = int(cfg["data"]["img_size"])
    batch = int(cfg["data"]["batch_size"])
    aug = cfg["data"]["aug"]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), reshuffle_each_iteration=True)

    def _map(path, label):
        img = _decode_image(path, img_size)
        if training:
            img = _augment(img, aug, img_size)
        return img, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds


def compute_class_weights(train_csv: str) -> dict[int, float]:
    """Gewichtung zum Ausgleich der Class-Imbalance (maligne stark unterrepräsentiert)."""
    df = pd.read_csv(train_csv)
    n = len(df)
    pos = int(df["label"].sum())
    neg = n - pos
    if pos == 0 or neg == 0:
        return {0: 1.0, 1: 1.0}
    # Inverse-Frequenz-Gewichtung (sklearn-Konvention: n / (2 * count))
    return {0: n / (2.0 * neg), 1: n / (2.0 * pos)}
