"""EfficientNet-Transfer-Learning-Modell (Schritt 3 des Plans).

Nutzt ein auf ImageNet vortrainiertes EfficientNet-B0 oder -B4 als Backbone und
ersetzt den Klassifikationskopf durch einen binären Ausgang (benigne vs. maligne).
Klein genug für spätere mobile Anwendung (TFLite), aber State-of-the-Art-Basis.

Wichtig: EfficientNet erwartet Eingaben in [0, 255]; die passende Normalisierung
steckt in `preprocess_input` und wird als erste Modell-Layer eingebettet, damit sie
nach dem TFLite-Export auch auf dem Gerät identisch angewendet wird.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras-2-Verhalten unter TF>=2.16

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

_BACKBONES = {
    "B0": (EfficientNetB0, 224),
    "B4": (EfficientNetB4, 380),
}


def build_model(cfg: dict[str, Any]) -> tf.keras.Model:
    name = cfg["model"]["backbone"].upper()
    if name not in _BACKBONES:
        raise ValueError(f"Unbekanntes Backbone '{name}', erlaubt: {list(_BACKBONES)}")
    backbone_fn, default_size = _BACKBONES[name]
    img_size = int(cfg["data"]["img_size"])

    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    # Normalisierung ins Modell einbetten -> identisch bei Training und On-Device-Inferenz
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)

    base = backbone_fn(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False  # Phase 1: eingefroren

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(float(cfg["model"]["dropout"]), name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="malignant_prob")(x)

    model = models.Model(inputs, outputs, name=f"skincancer_efficientnet_{name.lower()}")
    model.base_model = base  # Referenz zum Auftauen in Phase 2
    return model


def unfreeze_top(model: tf.keras.Model, n_layers: int) -> None:
    """Taut die obersten n Layer des Backbones fürs Fine-Tuning auf.

    BatchNorm-Layer bleiben eingefroren — das stabilisiert das Fine-Tuning bei
    kleinen medizinischen Datensätzen deutlich.
    """
    base = model.base_model
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
