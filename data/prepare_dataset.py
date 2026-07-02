"""Schritt 3 — Patienten-getrennter Train/Val/Test-Split.

Verhindert Data Leakage: Bilder desselben Patienten (patient_id) dürfen nicht
gleichzeitig in Trainings- und Testdaten liegen. Sonst würde auswendig gelerntes
Wissen als gute Testleistung fehlinterpretiert (Overfitting).

Schreibt drei Manifest-CSVs (train/val/test) mit Spalten: filepath, label, patient_id.

Nutzung:
    python data/prepare_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def grouped_split(df: pd.DataFrame, cfg: dict[str, Any]):
    s = cfg["split"]
    group_col = s["group_field"]

    # Fehlende patient_id: jede Zeile als eigene Gruppe behandeln (konservativ).
    groups = df[group_col].fillna("__na__" + df.index.astype(str))

    # 1) Test abtrennen
    gss1 = GroupShuffleSplit(n_splits=1, test_size=s["test_size"],
                             random_state=s["random_state"])
    trainval_idx, test_idx = next(gss1.split(df, df["label"], groups))
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # 2) Val aus dem Rest abtrennen
    groups_tv = groups.iloc[trainval_idx].reset_index(drop=True)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=s["val_size"],
                             random_state=s["random_state"])
    train_idx, val_idx = next(gss2.split(trainval, trainval["label"], groups_tv))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    # Sicherheitscheck: keine Patienten-Überschneidung zwischen den Splits
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pa = set(locals()[a][group_col].dropna())
        pb = set(locals()[b][group_col].dropna())
        overlap = pa & pb
        assert not overlap, f"Patienten-Leakage zwischen {a} und {b}: {overlap}"

    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Val/Test-Split (patientengetrennt)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["paths"]["filtered_csv"])

    cols = ["filepath", "label", "patient_id"]
    df = df[[c for c in cols if c in df.columns]].dropna(subset=["filepath", "label"])

    train, val, test = grouped_split(df, cfg)

    out_dir = cfg["paths"]["splits_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(out_dir, f"{name}.csv")
        part.to_csv(path, index=False)
        pos = int(part["label"].sum())
        print(f"{name:5s}: {len(part):5d} Bilder | maligne={pos} "
              f"({pos / max(len(part), 1):.1%}) -> {path}")


if __name__ == "__main__":
    main()
