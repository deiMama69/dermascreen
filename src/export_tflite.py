"""Schritt 5 — Export des trainierten Modells nach TensorFlow Lite.

Erzeugt eine .tflite-Datei für die On-Device-Inferenz in der Android-App.
Unterstützt float16-Quantisierung (halbe Größe, kaum Genauigkeitsverlust, gute
Wahl fürs Handy) oder int8-Quantisierung (kleinste/schnellste Variante, benötigt
ein repräsentatives Kalibrierungs-Dataset).

Der Betriebs-Schwellenwert aus der Evaluierung wird zusätzlich in eine JSON-Datei
geschrieben, damit die App bei genau diesem Wert „verdächtig" meldet.

Nutzung:
    python src/export_tflite.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from dataset import _decode_image  # gleicher Bild-Decode wie im Training


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def representative_dataset(cfg: dict[str, Any]):
    """Generator mit echten Trainingsbildern zur int8-Kalibrierung."""
    train_csv = os.path.join(cfg["paths"]["splits_dir"], "train.csv")
    full = pd.read_csv(train_csv)
    n = min(int(cfg["export"]["rep_dataset_size"]), len(full))
    df = full.sample(n=n, random_state=0)
    img_size = int(cfg["data"]["img_size"])
    for path in df["filepath"].astype(str):
        img = _decode_image(tf.constant(path), img_size)
        yield [tf.expand_dims(img, 0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="TFLite-Export")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    ckpt = os.path.join(cfg["paths"]["models_dir"], "best_model.keras")
    model = tf.keras.models.load_model(ckpt, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant = cfg["export"]["quantization"].lower()

    if quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(cfg)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    # quant == "none": Standard-Float32

    tflite_model = converter.convert()

    out_dir = cfg["paths"]["models_dir"]
    tflite_path = os.path.join(out_dir, "skincancer.tflite")
    with open(tflite_path, "wb") as fh:
        fh.write(tflite_model)
    size_mb = len(tflite_model) / 1e6
    print(f"TFLite-Modell ({quant}) gespeichert -> {tflite_path} ({size_mb:.1f} MB)")

    # Betriebs-Schwellenwert + Labels als Beipack-Metadaten für die App
    report_path = os.path.join(out_dir, "evaluation_report.json")
    threshold = 0.5
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            threshold = json.load(fh).get("operating_threshold", 0.5)

    meta = {
        "input_size": int(cfg["data"]["img_size"]),
        "operating_threshold": threshold,
        "labels": {"0": "benigne (unauffällig)", "1": "verdächtig / maligne"},
        "fitzpatrick_scope": cfg["filter"]["fitzpatrick_keep"],
        "disclaimer": "Kein Medizinprodukt. Keine Diagnose. Nur Forschungs-/Demozweck.",
    }
    meta_path = os.path.join(out_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Modell-Metadaten -> {meta_path}")

    # In die App-Assets kopieren, falls der Ordner existiert
    assets = os.path.join("android", "app", "src", "main", "assets")
    if os.path.isdir(assets):
        shutil.copy(tflite_path, os.path.join(assets, "skincancer.tflite"))
        shutil.copy(meta_path, os.path.join(assets, "model_meta.json"))
        print(f"Modell + Metadaten nach {assets} kopiert.")


if __name__ == "__main__":
    main()
