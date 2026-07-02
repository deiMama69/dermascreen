"""Schritt 2 — Filterung + binäres Label für die ISIC-Collection-406-Daten.

Wählt aus den lokalen Metadaten (data/raw/metadata.csv):
  * klinische Aufnahmen (image_type enthält "clinical" — Dermatoskop-Bilder raus),
  * eindeutiges Label aus diagnosis_1 ("Benign"/"Malignant"; "Indeterminate"/leer raus),
und baut das binäre Ziel-Label (1 = maligne, 0 = benigne). Die Bilder liegen nach
Schritt 1 bereits lokal — es wird nichts mehr heruntergeladen.

Ergebnis: data/raw/metadata_filtered.csv mit filepath, label, patient_id, lesion_id, ...

Nutzung:
    python data/filter_metadata.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_filters(df: pd.DataFrame, cfg: dict[str, Any], images_dir: str) -> pd.DataFrame:
    f = cfg["filter"]
    df = df.copy()

    label_col = f["label_field"]
    id_col = f["image_id_field"]
    itype_col = f["image_type_field"]

    def norm(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip()

    # --- Label aus diagnosis_1 ---
    diag = norm(df[label_col]).str.lower()
    mal = [v.lower() for v in f["malignant_values"]]
    ben = [v.lower() for v in f["benign_values"]]
    mask_label = diag.isin(mal + ben)

    # --- nur klinische Aufnahmen ---
    sub = f["image_type_keep_substring"].lower()
    mask_clinical = norm(df[itype_col]).str.lower().str.contains(sub, na=False)

    out = df[mask_label & mask_clinical].copy()
    out["label"] = diag[out.index].isin(mal).astype(int)

    # --- Dateipfad je Bild (img_id enthält die Endung, z. B. .jpg) ---
    out["filepath"] = out[id_col].astype(str).apply(
        lambda name: os.path.join(images_dir, name))
    before = len(out)
    out = out[out["filepath"].apply(os.path.exists)].copy()

    print("Filter-Ergebnis:")
    print(f"  eindeutiges Label     : {int(mask_label.sum())}")
    print(f"  klinische Aufnahmen   : {int(mask_clinical.sum())}")
    print(f"  -> beide erfüllt      : {before}")
    if before != len(out):
        print(f"  Bild-Datei fehlt      : {before - len(out)} verworfen")
    print(f"  -> nutzbar            : {len(out)}")
    if len(out):
        pos = int(out["label"].sum())
        print(f"     maligne={pos}, benigne={len(out) - pos}")
        print("\n  diagnosis_1-Verteilung (gefiltert):")
        print(diag[out.index].value_counts().to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ISIC-406-Daten filtern + Label bauen")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    df = pd.read_csv(cfg["paths"]["metadata_csv"])
    images_dir = os.path.join(cfg["paths"]["raw_dir"], "images")
    filtered = apply_filters(df, cfg, images_dir)

    if filtered.empty:
        raise SystemExit(
            "Keine Einträge nach Filterung. Prüfe filter in config.yaml "
            "(Spaltennamen / Label-Werte).")

    keep_cols = ["filepath", "label", "patient_id", "lesion_id",
                 cfg["filter"]["label_field"], cfg["filter"]["image_type_field"]]
    filtered = filtered[[c for c in keep_cols if c in filtered.columns]]

    out_csv = cfg["paths"]["filtered_csv"]
    filtered.to_csv(out_csv, index=False)
    print(f"\nGefilterte Metadaten gespeichert -> {out_csv}")


if __name__ == "__main__":
    main()
