"""Schritt 4 — Klinische Evaluierung auf dem strikt getrennten Test-Set.

Fokus auf Sensitivität (Recall) für maligne Läsionen statt auf Accuracy:
Ein übersehenes Melanom (False Negative) ist gefährlicher als ein Fehlalarm
(False Positive). Der Betriebs-Schwellenwert wird deshalb so gewählt, dass eine
Ziel-Sensitivität (config: target_sensitivity) erreicht wird, und die dabei
resultierende Spezifität wird berichtet.

Ausgaben: ROC-AUC, Konfusionsmatrix beim kalibrierten Schwellenwert, Sensitivität,
Spezifität, PPV/NPV sowie ein ROC-Plot.

Nutzung:
    python src/evaluate.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras-2-Verhalten unter TF>=2.16
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve)

from dataset import build_dataset
from model import build_model


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def collect_predictions(model, ds) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0).ravel()
        y_prob.append(p)
        y_true.append(y.numpy().ravel())
    return np.concatenate(y_true), np.concatenate(y_prob)


def threshold_for_sensitivity(y_true, y_prob, target: float) -> float:
    """Kleinster Schwellenwert, bei dem die Sensitivität >= target bleibt."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return 0.5
    # unter allen Schwellen mit ausreichender Sensitivität die mit bester Spezifität
    best = ok[np.argmax(1 - fpr[ok])]
    return float(thr[best])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluierung auf dem Test-Set")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Architektur im Code neu bauen und nur die Gewichte laden. Das umgeht den
    # tf-keras-Bug beim Speichern/Laden der EfficientNet-Normalization-Schicht im
    # .keras-Vollmodell-Format (die ImageNet-Normalisierungswerte kommen korrekt
    # aus dem frisch gebauten Backbone).
    ckpt = os.path.join(cfg["paths"]["models_dir"], "best_model.weights.h5")
    model = build_model(cfg)
    model.load_weights(ckpt)

    test_csv = os.path.join(cfg["paths"]["splits_dir"], "test.csv")
    test_ds = build_dataset(test_csv, cfg, training=False)

    y_true, y_prob = collect_predictions(model, test_ds)
    auc = roc_auc_score(y_true, y_prob)

    target = float(cfg["evaluate"]["target_sensitivity"])
    thr = threshold_for_sensitivity(y_true, y_prob, target)
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    report = {
        "roc_auc": round(float(auc), 4),
        "operating_threshold": round(thr, 4),
        "target_sensitivity": target,
        "sensitivity_recall_malignant": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv_precision": round(ppv, 4),
        "npv": round(npv, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    out = os.path.join(cfg["paths"]["models_dir"], "evaluation_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nBericht gespeichert -> {out}")

    if fn > 0:
        print(f"\n⚠️  {fn} maligne Läsion(en) übersehen (False Negatives). "
              f"Beim Einsatz ist jeder FN kritisch — Schwellenwert ggf. senken.")

    # ROC-Plot (optional, nur wenn matplotlib verfügbar)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("1 − Spezifität (FPR)")
        plt.ylabel("Sensitivität (TPR)")
        plt.title("ROC — Test-Set")
        plt.legend()
        plot_path = os.path.join(cfg["paths"]["models_dir"], "roc_curve.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        print(f"ROC-Plot gespeichert -> {plot_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
