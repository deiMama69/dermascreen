"""Schritt 2 — Filterung nach Fitzpatrick I–III + klinischen Aufnahmen, dann Download.

Wählt aus den Rohmetadaten gezielt:
  * Fitzpatrick-Hauttypen I, II, III  (helle Haut),
  * klinische Makroaufnahmen (image_type enthält "clinical") — KEINE Dermatoskop-Bilder,
  * Einträge mit eindeutigem benign/malignant-Label.

Lädt anschließend nur die Bilder dieser gefilterten Einträge herunter und schreibt
ein bereinigtes CSV (metadata_filtered.csv) mit lokalem Dateipfad + binärem Label.

Nutzung:
    python data/filter_metadata.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_filters(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    f = cfg["filter"]

    # Feld-Werte robust normalisieren (Groß/Kleinschreibung, Whitespace)
    def norm(series: pd.Series) -> pd.Series:
        return series.astype("string").str.strip()

    df = df.copy()
    df["fitzpatrick_skin_type"] = norm(df["fitzpatrick_skin_type"])
    df["image_type"] = norm(df["image_type"]).str.lower()
    df[f["label_field"]] = norm(df[f["label_field"]]).str.lower()

    # 1) Fitzpatrick I–III
    keep_fitz = [x.strip() for x in f["fitzpatrick_keep"]]
    mask_fitz = df["fitzpatrick_skin_type"].isin(keep_fitz)

    # 2) Nur klinische Aufnahmen (Dermatoskop-Bilder verwerfen)
    sub = f["image_type_keep_substring"].lower()
    mask_clinical = df["image_type"].str.contains(sub, na=False)

    # 3) Eindeutiges Label
    mal = [v.lower() for v in f["malignant_values"]]
    ben = [v.lower() for v in f["benign_values"]]
    mask_label = df[f["label_field"]].isin(mal + ben)

    out = df[mask_fitz & mask_clinical & mask_label].copy()

    # Binäres Ziel-Label: 1 = maligne, 0 = benigne
    out["label"] = out[f["label_field"]].isin(mal).astype(int)

    print("Filter-Ergebnis:")
    print(f"  Fitzpatrick I–III     : {mask_fitz.sum()}")
    print(f"  klinische Aufnahmen   : {mask_clinical.sum()}")
    print(f"  eindeutiges Label     : {mask_label.sum()}")
    print(f"  -> alle drei erfüllt  : {len(out)}")
    print(f"     davon maligne={int(out['label'].sum())}, "
          f"benigne={int((out['label'] == 0).sum())}")
    return out


def download_images(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    raw_dir = cfg["paths"]["raw_dir"]
    img_dir = os.path.join(raw_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    local_paths: list[str | None] = []
    session = requests.Session()
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Bild-Download"):
        isic_id = getattr(row, "isic_id")
        url = getattr(row, "url")
        dest = os.path.join(img_dir, f"{isic_id}.jpg")
        if os.path.exists(dest):
            local_paths.append(dest)
            continue
        if not isinstance(url, str) or not url:
            local_paths.append(None)
            continue
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(r.content)
            local_paths.append(dest)
        except requests.RequestException:
            local_paths.append(None)

    df = df.copy()
    df["filepath"] = local_paths
    before = len(df)
    df = df[df["filepath"].notna()].reset_index(drop=True)
    print(f"{len(df)}/{before} Bilder erfolgreich geladen.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="ISIC-Metadaten filtern + Bilder laden")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-download", action="store_true",
                        help="Nur filtern, keine Bilder herunterladen")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg["paths"]["metadata_csv"])
    filtered = apply_filters(df, cfg)

    if filtered.empty:
        raise SystemExit("Keine Einträge nach Filterung. Prüfe die Filter in config.yaml.")

    if not args.no_download:
        filtered = download_images(filtered, cfg)

    out_csv = cfg["paths"]["filtered_csv"]
    filtered.to_csv(out_csv, index=False)
    print(f"Gefilterte Metadaten gespeichert -> {out_csv}")


if __name__ == "__main__":
    main()
