"""Schritt 1 — Download der ISIC-Metadaten und -Bilder über die REST-API.

Lädt seitenweise Metadaten aus dem ISIC-Archiv (api.isic-archive.com/api/v2) und
speichert sie als CSV. Die eigentlichen Bilder werden erst in Schritt 2
(filter_metadata.py) heruntergeladen — nur für die Einträge, die den Fitzpatrick-
und Klinik-Filter bestehen. So wird kein Speicher für unbrauchbare Dermatoskop-
Bilder verschwendet.

Nutzung:
    python data/download_isic.py --config config.yaml

Hinweis: Für den vollständigen Zugriff empfiehlt sich ein kostenloser ISIC-Account.
Der anonyme Zugriff ist rate-limitiert. Alternativ kann das offizielle `isic-cli`
verwendet werden (siehe README).
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


def _flatten_record(item: dict[str, Any]) -> dict[str, Any]:
    """Zieht die relevanten, verschachtelten Metadaten-Felder flach in eine Zeile."""
    meta = item.get("metadata", {}) or {}
    clinical = meta.get("clinical", {}) or {}
    acquisition = meta.get("acquisition", {}) or {}
    files = item.get("files", {}) or {}
    full = files.get("full", {}) or {}
    return {
        "isic_id": item.get("isic_id"),
        "url": full.get("url"),
        "image_type": acquisition.get("image_type"),
        "fitzpatrick_skin_type": clinical.get("fitzpatrick_skin_type"),
        "benign_malignant": clinical.get("benign_malignant"),
        "diagnosis": clinical.get("diagnosis"),
        "patient_id": clinical.get("patient_id"),
        "lesion_id": clinical.get("lesion_id"),
        "anatom_site": clinical.get("anatom_site_general"),
        "age": clinical.get("age_approx"),
        "sex": clinical.get("sex"),
    }


def fetch_metadata(cfg: dict[str, Any]) -> pd.DataFrame:
    api_base = cfg["isic"]["api_base"].rstrip("/")
    limit = int(cfg["isic"]["page_limit"])
    max_images = int(cfg["isic"]["max_images"])
    token = cfg["isic"].get("token") or ""

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url: str | None = f"{api_base}/images/?limit={limit}"
    rows: list[dict[str, Any]] = []

    with tqdm(total=max_images, desc="ISIC-Metadaten", unit="img") as bar:
        while url and len(rows) < max_images:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 429:  # rate limit
                time.sleep(5)
                continue
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("results", []):
                rows.append(_flatten_record(item))
                bar.update(1)
            url = payload.get("next")  # Cursor-Pagination

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISIC-Metadaten herunterladen")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_csv = cfg["paths"]["metadata_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = fetch_metadata(cfg)
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} Metadaten-Einträge gespeichert -> {out_csv}")
    # Kurze Übersicht der Verteilungen zur Kontrolle
    for col in ("image_type", "fitzpatrick_skin_type", "benign_malignant"):
        print(f"\nVerteilung '{col}':")
        print(df[col].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
