
# LLava Installation und Startanleitung

## Voraussetzungen
- Python 3.10
- pip
- virtualenv
- cuda

## Installation

1. **Virtuelle Umgebung erstellen**

   Erstellen Sie eine neue virtuelle Umgebung namens `robodoc_llava`:

   ```bash
   virtualenv robodoc_llava
   ```

2. **Virtuelle Umgebung aktivieren**

   Aktivieren Sie die virtuelle Umgebung:

   ```bash
   source robodoc_llava/bin/activate
   ```

3. **Abhängigkeiten installieren**

   Installieren Sie die notwendigen Pakete mit pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Hugging Face Token konfigurieren**

   In der `config.json` ersetzen Sie `TOKEN_HERE` in der mit Ihrem tatsächlichen Hugging Face Token.

   ```json
   {"HF_TOKEN":"TOKEN_HERE"}
   ```
   

5. **Anwendung starten**

   Starten Sie die Anwendung durch Ausführen der `setup.py` Datei:

   ```bash
   python setup.py
   ```
      

## Nutzung

Nachdem Sie LLava installiert und die `setup.py` ausgeführt haben, sollte die Anwendung laufen und bereit für die Nutzung sein, die Datei `llava_inference.py` stelle dabei die Schnittstelle dar, welche von Applikationen zum Abrufen verwendet wird.

Die Erste Ausführung kann einige Zeit dauern, da das Modell zunächst heruntergeladen und gespeichert werden muss, anschließend wird es nur noch zum Bearbeiten von Anfragen in den VRAM geladen und anschließend unloaded.
