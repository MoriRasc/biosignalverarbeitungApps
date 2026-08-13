# Biosignalverarbeitung Apps

Dieses Repository enthält eine Sammlung interaktiver Streamlit-Apps zur Vermittlung zentraler Konzepte der digitalen Signalverarbeitung und Biosignalverarbeitung. Die Anwendungen sind bewusst visualisierend aufgebaut, damit Frequenzinhalte, Abtastung, Filterung und Rekonstruktion anschaulich nachvollzogen werden können.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Details finden Sie in der Datei [LICENSE](LICENSE).

## Überblick

Die Apps decken typische Themen aus der Signalverarbeitung ab:

- Abtastung und Nyquist-Theorem
- Dezimierung und Interpolation
- Downsampling und Upsampling
- Filterung im Zeit- und Frequenzbereich
- Fourier-Reihen und harmonische Synthese

## Enthaltene Anwendungen

### 1. Abtasttheorem-App
Datei: `abtasttheoremApp.py`

Erklärt das Abtasttheorem anhand eines sinusförmigen Signals. Der Nutzer kann die Abtastfrequenz variieren und sofort beobachten, wann Aliasing auftritt und wie das Signal rekonstruiert wird.

### 2. Dezimierung und Interpolation
Datei: `dezimierungInterpolation.py`

Zeigt, wie ein Signal durch Dezimierung und Interpolation verändert wird und welche Auswirkungen diese Vorgänge auf das Zeit- und Frequenzspektrum haben. Die App veranschaulicht die Folgen einer zu niedrigen Abtastfrequenz und die Rolle der Resampling-Operationen.

### 3. Filterungs-App
Datei: `filterBasicsApp.py`

Demonstriert Tiefpass-, Hochpass-, Bandpass- und Bandsperren-Filter. Nutzer können Frequenzkomponenten, Filterordnung und Grenzfrequenzen anpassen und den Effekt im Zeit- und Frequenzbereich beobachten.

### 4. Fourier Explorer
Datei: `fourier_explorer.py`

Ein didaktisches Werkzeug zur Darstellung der Fourier-Reihen. Es visualisiert die Approximation von Rechteck-, Sägezahn- und Dreieckswellen sowie die Zerlegung von Audio-Signalen in Frequenzanteile.

### 5. Weitere didaktische Anwendungen
Datei: `freqResolutionApp.py`

Beschäftigt sich mit der Frequenzauflösung und zeigt, wie die Wahl der Messdauer und der Abtastfrequenz die Auflösung im Spektrum beeinflusst.

Datei: `quantisierungApp.py`

Veranschaulicht Quantisierungseffekte und zeigt, wie eine digitale Abbildung von Analogsignalen zu Fehlern führt.

## Technischer Stack

- Python
- Streamlit
- NumPy
- SciPy
- Matplotlib
- Plotly

## Voraussetzungen

Es wird Python 3.10+ empfohlen.

## Installation

1. Repository klonen
2. Virtuelle Umgebung anlegen und aktivieren
3. Abhängigkeiten installieren

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# oder bash/zsh
# source .venv/bin/activate

pip install -r requirements.txt
```

## Ausführen der Apps

Jede App kann einzeln gestartet werden:

```bash
streamlit run abtasttheoremApp.py
streamlit run dezimierungInterpolation.py
streamlit run filterBasicsApp.py
streamlit run fourier_explorer.py
streamlit run freqResolutionApp.py
streamlit run quantisierungApp.py
```

Alternativ mit Python-Modulsyntax:

```bash
python -m streamlit run abtasttheoremApp.py
```

## Lernziel

Das Projekt dient als interaktive Lernumgebung für Grundlagen der Biosignalverarbeitung und digitalen Signalverarbeitung. Es eignet sich besonders für:

- Vorlesungen und Übungen
- Demonstrationen zu Sampling und FFT
- Visualisierung von Filter- und Aliasing-Effekten
- Verständnis der Beziehung zwischen Zeitbereich und Frequenzbereich

## Projektstruktur

```text
biosignalverarbeitungApps/
├── README.md
├── abtasttheoremApp.py
├── dezimierungInterpolation.py
├── filterBasicsApp.py
├── fourier_explorer.py
├── freqResolutionApp.py
├── quantisierungApp.py
├── requirements.txt
└── .venv/ (optional)
```

## Hinweise

- Die Apps sind primär didaktisch und nicht als produktionsreife Analyse-Tools konzipiert.
- Für die besten Ergebnisse wird empfohlen, die einzelnen Anwendungen separat zu starten und mit den Slidern zu experimentieren.
- Einige Apps verwenden zusätzliche Audio- oder Signalbeispiele, die durch Uploads oder interne Beispieldaten ersetzt werden können.

## Weiterführende Themen

- Diskrete Zeit- und Fourier-Transformation
- Digitales Signal- und Biosignal-Processing
- Wavelet-Analyse
- Filterdesign und Signalrekonstruktion
- Messsysteme und Abtastfehler
