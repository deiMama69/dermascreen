# Loslegen — Schritt-für-Schritt (für Einsteiger)

Diese Anleitung führt dich von Null bis zum fertigen Modell in der App. Du musst
**nicht** programmieren können — du tippst nur die angegebenen Befehle ab.

> ⚠️ **Wichtig zuerst:** Das ist ein Lern-/Demo-Projekt, **kein Medizinprodukt**.
> Nichts, was hier herauskommt, ist eine Diagnose. Bei jeder auffälligen Hautstelle
> gehört der Gang zur Hautärztin/zum Hautarzt — unabhängig vom Ergebnis.

---

## Welchen Weg soll ich nehmen?

Es gibt zwei Wege. **Fang mit Weg A an** — er ist für Einsteiger deutlich einfacher
und du hast am schnellsten ein Erfolgserlebnis.

| | **Weg A: Google Colab** (empfohlen) | **Weg B: Lokal auf deinem PC** |
|---|---|---|
| Installation nötig? | Nein (läuft im Browser) | Ja (Python + Pakete) |
| Geschwindigkeit | Schnell (kostenlose GPU) | Langsam (nur CPU — deine AMD-GPU wird von TensorFlow nicht genutzt) |
| Gut für | den ersten kompletten Durchlauf | offline arbeiten, alles bleibt auf dem Rechner |

Du kannst später jederzeit auf Weg B wechseln. Beide erzeugen dieselbe `.tflite`-Datei
für die App.

---

## Weg A — Google Colab (der einfache Weg)

1. Kostenlosen Google-Account haben (Gmail reicht).
2. Öffne <https://colab.research.google.com> → oben **Datei → Notebook hochladen** →
   wähle `notebooks/colab_training.ipynb` aus diesem Projektordner.
3. Oben im Menü **Laufzeit → Laufzeittyp ändern → Hardwarebeschleuniger: GPU (T4)** →
   Speichern.
4. Lade die Projektdateien ins Colab: linke Seitenleiste (Ordner-Symbol) → per
   Drag-and-Drop den Inhalt von `SkinCancerAI` hineinziehen. (Einfacher, wenn du das
   Projekt auf GitHub legst und in der ersten Zelle klonst — aber das ist optional.)
5. Führe die Zellen **von oben nach unten** aus: auf jede Zelle klicken und
   `Shift + Enter` drücken. Warte, bis eine Zelle fertig ist, bevor du die nächste
   startest.
6. Die letzte Zelle lädt automatisch `skincancer.tflite` und `model_meta.json`
   herunter. Diese zwei Dateien brauchst du für die App (siehe unten „App").

Das war's. Wenn du das geschafft hast, ist der Rest Kür.

---

## Weg B — Lokal auf deinem PC (CPU)

### Schritt 0 — Was dich erwartet
Auf deinem PC läuft das Training nur auf dem Prozessor (deine AMD-Grafikkarte kann
TensorFlow unter Windows nicht nutzen). Das ist langsamer, aber machbar, weil wir das
kleine Modell EfficientNet-B0 verwenden. Rechne mit **Minuten pro Epoche**; ein
kompletter Lauf kann über Nacht gehen. Lass den PC dabei nicht in den Ruhezustand
gehen (Windows-Einstellungen → System → Netzbetrieb & Akku → Ruhezustand: „Nie" beim
Netzbetrieb).

### Schritt 1 — Python installieren
1. Gehe auf <https://www.python.org/downloads/> und lade **Python 3.11** herunter
   (nicht 3.12 — damit gibt es weniger Probleme).
2. Installer starten. **Ganz wichtig:** unten im ersten Fenster den Haken bei
   **„Add python.exe to PATH"** setzen, dann auf **Install Now**.

### Schritt 2 — Windows-Platzhalter abschalten
Windows hat einen „Fake"-Python-Eintrag, der stört. Abschalten:
- **Einstellungen → Apps → Erweiterte App-Einstellungen → App-Ausführungsaliase**
- Die Schalter für **`python.exe`** und **`python3.exe`** auf **Aus** stellen.

### Schritt 3 — Prüfen, ob Python läuft
Öffne **PowerShell** (Startmenü → „PowerShell" tippen → Enter) und gib ein:
```powershell
py --version
```
Es sollte etwas wie `Python 3.11.x` erscheinen. Wenn ja: super.

### Schritt 4 — Ins Projekt wechseln
```powershell
cd C:\SkinCancerAI
```

### Schritt 5 — Isolierte Umgebung anlegen und aktivieren
Das hält die Pakete dieses Projekts getrennt vom Rest des Systems.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Danach steht vorne in der Zeile `(.venv)`. **Falls Fehler** „... kann nicht geladen
werden, da die Ausführung von Skripts deaktiviert ist", einmalig ausführen und dann
Schritt 5 wiederholen:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
(Bei der Nachfrage `J` eingeben.)

### Schritt 6 — Benötigte Pakete installieren
```powershell
pip install -r requirements.txt
```
Das lädt TensorFlow & Co. — dauert ein paar Minuten und ~1 GB Download. Einmalig.

### Schritt 7 — Daten holen (ISIC-Bilder)
Nacheinander ausführen — jeder Befehl erst starten, wenn der vorherige fertig ist:
```powershell
python data\download_isic.py   --config config.yaml
python data\filter_metadata.py --config config.yaml
python data\prepare_dataset.py --config config.yaml
```
- Der erste Befehl lädt die Metadaten (Beschreibungen aller Bilder).
- Der zweite filtert auf **helle Haut (Fitzpatrick I–III)** + **klinische Fotos** und
  lädt nur diese Bilder herunter.
- Der dritte teilt die Daten sauber in Trainings-, Validierungs- und Testteil.

**Schau dir die Ausgabe von Schritt 2 an:** Dort steht, wie viele maligne (bösartige)
und benigne (gutartige) Bilder übrig bleiben. Sind es sehr wenige maligne (z. B. unter
~50), ist das Ergebnis später nicht aussagekräftig — dann sag mir Bescheid, dann bauen
wir eine zusätzliche Datenquelle ein.

### Schritt 8 — Training starten
```powershell
python src\train.py --config config.yaml
```
Es laufen zwei Phasen (erst der neue „Kopf", dann Feinschliff). Während des Trainings
siehst du pro Epoche Zahlen wie `val_auc` (je näher an 1,0, desto besser). Am Ende
liegt das beste Modell unter `models\best_model.keras`.

### Schritt 9 — Qualität prüfen
```powershell
python src\evaluate.py --config config.yaml
```
Das testet auf Bildern, die das Modell **nie** gesehen hat. Achte vor allem auf
**„sensitivity_recall_malignant"** (wie viele echte bösartige Fälle erkannt werden) —
diese Zahl ist bei Hautkrebs wichtiger als die reine Trefferquote.

### Schritt 10 — Modell für die App exportieren
```powershell
python src\export_tflite.py --config config.yaml
```
Das erzeugt `models\skincancer.tflite` und kopiert es (samt Metadaten) automatisch in
den App-Ordner.

---

## Das Modell in die App bringen

1. Installiere **Android Studio** (kostenlos, von Google).
2. Öffne darin den Ordner `C:\SkinCancerAI\android`.
3. Prüfe, dass in `android\app\src\main\assets\` die Dateien `skincancer.tflite` und
   `model_meta.json` liegen (Schritt 10 bzw. Colab-Download legen sie dort ab).
4. Schließe dein Android-Handy per USB an (mit aktiviertem Entwicklermodus) und drücke
   in Android Studio auf **Run ▶**.

Details dazu stehen in `android\README.md`.

---

## Häufige Stolpersteine

- **`python` macht nichts / öffnet den Store:** Schritt 2 (Platzhalter abschalten)
  wurde übersprungen. Nachholen, PowerShell neu öffnen.
- **`pip` nicht gefunden:** Umgebung nicht aktiv — Schritt 5 (`Activate.ps1`) erneut
  ausführen; vorne muss `(.venv)` stehen.
- **Download bricht ab / ist langsam:** Der anonyme ISIC-Zugriff ist gedrosselt. Ein
  kostenloser Account auf <https://www.isic-archive.com> und der Token in `config.yaml`
  (`isic.token`) helfen.
- **Training „hängt":** Es hängt meist nicht, sondern rechnet nur langsam auf der CPU.
  Solange sich die Zahlen pro Epoche ändern, läuft alles.
- **PC geht schlafen und bricht ab:** Ruhezustand im Netzbetrieb deaktivieren
  (Schritt 0).

---

## Kurz gesagt — deine Reihenfolge
1. **Erst Weg A (Colab)** ausprobieren → schnellster erster Erfolg.
2. Wenn du es lokal willst: Weg B, Schritte 1–10 der Reihe nach.
3. Danach Modell in die App (Android Studio).

Wenn irgendwo eine Fehlermeldung kommt: kopier sie mir, dann sag ich dir genau, was zu
tun ist.
