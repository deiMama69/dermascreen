"""Schritt 1 — Download von PAD-UFES-20 über die ISIC-Collection 406.

Lädt die Metadaten aller Bilder der ISIC-Collection 406 (= PAD-UFES-20, klinische
Smartphone-Fotos) über den Such-Endpoint und danach die Bilddateien selbst.
Jedes Record liefert Label (diagnosis_1), patient_id und lesion_id direkt mit —
das ermöglicht später den sauberen Split auf Patientenebene.

Ergebnis:
  * data/raw/metadata.csv          — Tabelle mit img_id, diagnosis_1, patient_id, ...
  * data/raw/images/<isic_id>.jpg  — die Bilder

Nutzung:
    python data/download_isic406.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import pandas as pd
import requests
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _flatten(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata", {}) or {}
    acq = meta.get("acquisition", {}) or {}
    clin = meta.get("clinical", {}) or {}
    files = item.get("files", {}) or {}
    full = files.get("full", {}) or {}
    isic_id = item.get("isic_id")
    return {
        "img_id": f"{isic_id}.jpg",     # lokaler Dateiname
        "isic_id": isic_id,
        "url": full.get("url"),
        "image_type": acq.get("image_type"),
        "diagnosis_1": clin.get("diagnosis_1"),
        "diagnosis_2": clin.get("diagnosis_2"),
        "diagnosis_3": clin.get("diagnosis_3"),
        "patient_id": clin.get("patient_id"),
        "lesion_id": clin.get("lesion_id"),
        "anatom_site": clin.get("anatom_site_1"),
        "age": clin.get("age_approx"),
        "sex": clin.get("sex"),
    }


def fetch_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    api = cfg["isic"]["api_base"].rstrip("/")
    cid = int(cfg["isic"]["collection_id"])
    limit = int(cfg["isic"]["page_limit"])
    token = cfg["isic"].get("token") or ""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # WICHTIG: der Such-Endpoint (images/search/) respektiert den collections-Filter.
    url: str | None = f"{api}/images/search/?collections={cid}&limit={limit}"
    rows: list[dict[str, Any]] = []
    bar = tqdm(desc="ISIC-Metadaten", unit="img")
    while url:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code == 429:
            time.sleep(5)
            continue
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("results", []):
            rows.append(_flatten(item))
            bar.update(1)
        url = payload.get("next")
    bar.close()
    return pd.DataFrame(rows)


def download_images(df: pd.DataFrame, images_dir: str) -> int:
    os.makedirs(images_dir, exist_ok=True)
    session = requests.Session()
    ok = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Bild-Download"):
        dest = os.path.join(images_dir, getattr(row, "img_id"))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            ok += 1
            continue
        url = getattr(row, "url")
        if not isinstance(url, str) or not url:
            continue
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as fh:
                fh.write(r.content)
            ok += 1
        except requests.RequestException:
            pass
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="PAD-UFES-20 via ISIC-Collection 406 laden")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-images", action="store_true", help="nur Metadaten laden")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_dir = cfg["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    images_dir = os.path.join(raw_dir, "images")
    meta_csv = cfg["paths"]["metadata_csv"]

    df = fetch_metadata(cfg)
    df.to_csv(meta_csv, index=False)
    print(f"\n{len(df)} Metadaten-Einträge -> {meta_csv}")
    print("\nVerteilung 'image_type':")
    print(df["image_type"].value_counts(dropna=False).to_string())
    print("\nVerteilung 'diagnosis_1':")
    print(df["diagnosis_1"].value_counts(dropna=False).to_string())

    if not args.no_images:
        n = download_images(df, images_dir)
        print(f"\n{n}/{len(df)} Bilder geladen -> {images_dir}")


if __name__ == "__main__":
    main()
