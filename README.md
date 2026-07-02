# SkinCancerAI — Hautkrebs-Erkennung aus Alltagsfotos

Ein Transfer-Learning-Projekt (EfficientNet) zur Erkennung verdächtiger Hautläsionen
aus **normalen Kamera-/Smartphone-Fotos** (klinische Makroaufnahmen, **keine** Dermatoskop-Bilder).
Zugeschnitten auf helle Hauttypen (**Fitzpatrick I–III**). Deployment als **TensorFlow Lite**
in einer **Jetpack-Compose**-App mit On-Device-Inferenz.

> ⚠️ **Medizinischer Hinweis / Disclaimer**
> Dieses Projekt ist ausschließlich für Forschungs-, Lern- und Demonstrationszwecke.
> Es ist **kein Medizinprodukt** und **keine diagnostische Hilfe**. Die Ausgabe ersetzt
> **niemals** die Untersuchung durch eine Ärztin oder einen Arzt. Jede Hautveränderung,
> die auffällig ist, sich verändert oder Sorgen bereitet, muss dermatologisch abgeklärt
> werden — unabhängig davon, was das Modell anzeigt. Modelle, die auf öffentlichen
> Datensätzen trainiert wurden, generalisieren nachweislich schlecht auf neue Kameras,
> Beleuchtungen und Populationen.

## Pipeline im Überblick

| Schritt | Skript | Zweck |
|--------|--------|-------|
| 1. Download | `data/download_isic406.py` | PAD-UFES-20 via ISIC-Collection 406 laden (klinische Fotos) |
| 2. Filterung | `data/filter_metadata.py` | Klinische Aufnahmen + Label (benigne/maligne) selektieren |
| 3. Split | `data/prepare_dataset.py` | Patienten-getrennter Train/Val/Test-Split (kein Leakage) |
| 4. Datenpipeline | `src/dataset.py` | `tf.data`-Pipeline inkl. Smartphone-Augmentation |
| 5. Modell | `src/model.py` | EfficientNet-B0/B4 mit ImageNet-Gewichten |
| 6. Training | `src/train.py` | Zwei-Phasen-Fine-Tuning + Class-Weighting |
| 7. Evaluierung | `src/evaluate.py` | Sensitivität/Spezifität, ROC-AUC, Schwellenwert-Tuning |
| 8. Export | `src/export_tflite.py` | Konvertierung nach `.tflite` (float16 / int8) |
| 9. App | `android/` | Kotlin + Jetpack Compose + CameraX + TFLite |

## Schnellstart

```bash
# 1. Umgebung
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Daten holen (PAD-UFES-20 via ISIC-Collection 406, kein Account nötig)
python data/download_isic406.py --config config.yaml
python data/filter_metadata.py  --config config.yaml
python data/prepare_dataset.py  --config config.yaml

# 3. Training (GPU empfohlen — siehe notebooks/colab_training.ipynb für Colab)
python src/train.py --config config.yaml

# 4. Evaluierung auf strikt getrenntem Test-Set
python src/evaluate.py --config config.yaml

# 5. TFLite-Export für die App
python src/export_tflite.py --config config.yaml
```

## Wichtige Design-Entscheidungen

- **Nur klinische Bilder**: Der Filter verwirft dermatoskopische Aufnahmen, weil ein darauf
  trainiertes Modell an echten Smartphone-Fotos scheitert (Domain-Gap).
- **Patienten-getrennter Split**: Bilder desselben Patienten dürfen nicht gleichzeitig in
  Train und Test landen — sonst wird Overfitting als „gute" Testleistung fehlinterpretiert.
- **Sensitivität vor Accuracy**: Bei Krebs ist ein übersehenes Melanom (False Negative) weit
  schlimmer als ein Fehlalarm. Der Betriebs-Schwellenwert wird auf eine Ziel-Sensitivität
  (Standard: 0,95) kalibriert, nicht auf maximale Accuracy.
- **Fitzpatrick I–III**: Das Modell wird bewusst auf helle Hauttypen eingegrenzt. Auf
  dunkleren Hauttypen (IV–VI) ist es **nicht** validiert und darf dort nicht eingesetzt werden.

## Bekannte Grenzen

- Klinische ISIC-Aufnahmen sind deutlich seltener als dermatoskopische — der nutzbare
  Datensatz ist klein. Ergebnisse sind ohne zusätzliche/externe Validierung nicht belastbar.
- Datensatz-Bias (Herkunft der Bilder, Kamera, Population) überträgt sich auf das Modell.
- Keine Zulassung, keine klinische Validierung, kein CE-/FDA-Status.
