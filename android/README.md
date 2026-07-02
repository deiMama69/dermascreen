# SkinCancerAI — Android-App (Jetpack Compose + CameraX + TFLite)

On-Device-Inferenz für die Hautkrebs-Demo. Alles läuft lokal — **keine Fotos werden
hochgeladen**. App-Ergebnisse sind **keine Diagnose** (siehe Disclaimer in der App).

## Einrichtung in Android Studio

1. Neues Projekt in Android Studio (Giraffe+), oder dieses `android/`-Verzeichnis als
   Modul öffnen. Die Kernklassen liegen unter
   `app/src/main/java/com/skincancerai/app/`.
2. Trainiertes Modell einbinden: `src/export_tflite.py` kopiert `skincancer.tflite`
   und `model_meta.json` automatisch nach `app/src/main/assets/`. Alternativ die
   beiden Dateien manuell dorthin legen.
3. Gradle-Sync ausführen. Benötigte Abhängigkeiten stehen in `app/build.gradle.kts`
   (Compose BOM, CameraX, TensorFlow Lite).
4. Auf einem echten Gerät starten (Kamera). Beim ersten Start Kamerazugriff erlauben.

## Aufbau

| Datei | Zweck |
|-------|-------|
| `MainActivity.kt` | Compose-UI, CameraX-Vorschau, Aufnahme, Ergebnis-Anzeige |
| `SkinClassifier.kt` | Lädt `.tflite` + `model_meta.json`, führt die Inferenz aus |
| `assets/skincancer.tflite` | Das exportierte Modell (aus dem Python-Training) |
| `assets/model_meta.json` | Eingabegröße, Betriebs-Schwellenwert, Labels, Disclaimer |

## Wichtige Details

- **Normalisierung im Modell**: `preprocess_input` ist als Layer eingebettet, daher
  übergibt die App rohe RGB-Werte in `[0,255]`. Kein separates Preprocessing nötig.
- **Schwellenwert**: `SkinClassifier` liest den in der Evaluierung auf hohe
  Sensitivität kalibrierten Schwellenwert aus `model_meta.json` (nicht hart 0,5).
- **`noCompress "tflite"`** in `build.gradle.kts` erlaubt das memory-mapped Laden.

## Grenzen

Nur für Fitzpatrick I–III als Demo trainiert. Nicht auf dunkleren Hauttypen validiert,
nicht zugelassen, kein Medizinprodukt.
