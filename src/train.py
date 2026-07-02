"""Schritt 3 — Zwei-Phasen-Training mit Class-Weighting.

Phase 1: Backbone eingefroren, nur der neue Kopf lernt (schnell, stabil).
Phase 2: Obere Backbone-Layer auftauen, mit sehr kleiner Lernrate feinjustieren.

Class-Imbalance wird über class_weight ausgeglichen (maligne Klasse höher gewichtet),
damit das Modell seltene, aber kritische maligne Fälle nicht ignoriert.
Der beste Checkpoint wird nach val-AUC gespeichert — nicht nach Accuracy, die bei
starker Imbalance trügt.

Nutzung:
    python src/train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras-2-Verhalten unter TF>=2.16
from typing import Any

import tensorflow as tf
import yaml

from dataset import build_dataset, compute_class_weights
from model import build_model, unfreeze_top


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _metrics() -> list:
    return [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall"),        # = Sensitivität (maligne)
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    ]


def _callbacks(cfg: dict[str, Any], ckpt_path: str) -> list:
    monitor = cfg["train"]["monitor"]
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor=monitor, mode="max",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max",
            patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.3, patience=3, verbose=1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Training des Hautkrebs-Modells")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    splits = cfg["paths"]["splits_dir"]
    train_csv = os.path.join(splits, "train.csv")
    val_csv = os.path.join(splits, "val.csv")

    train_ds = build_dataset(train_csv, cfg, training=True)
    val_ds = build_dataset(val_csv, cfg, training=False)

    class_weight = compute_class_weights(train_csv) if cfg["train"]["use_class_weight"] else None
    if class_weight:
        print(f"Class-Weights: {class_weight}")

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=cfg["train"]["label_smoothing"])

    model = build_model(cfg)
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    ckpt = os.path.join(cfg["paths"]["models_dir"], "best_model.weights.h5")

    # ---- Phase 1: nur Kopf ----
    print("\n=== Phase 1: Kopf-Training (Backbone eingefroren) ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["train"]["head_lr"]),
                  loss=loss, metrics=_metrics())
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg["train"]["head_epochs"],
              class_weight=class_weight,
              callbacks=_callbacks(cfg, ckpt))

    # ---- Phase 2: Fine-Tuning ----
    print("\n=== Phase 2: Fine-Tuning (obere Backbone-Layer aufgetaut) ===")
    unfreeze_top(model, int(cfg["train"]["finetune_unfreeze"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["train"]["finetune_lr"]),
                  loss=loss, metrics=_metrics())
    model.fit(train_ds, validation_data=val_ds,
              epochs=cfg["train"]["finetune_epochs"],
              class_weight=class_weight,
              callbacks=_callbacks(cfg, ckpt))

    print(f"\nBestes Modell gespeichert -> {ckpt}")
    print("Nächster Schritt: python src/evaluate.py --config config.yaml")


if __name__ == "__main__":
    main()
