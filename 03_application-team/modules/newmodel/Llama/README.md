
# LLama3 Installation und Startanleitung

## Voraussetzungen
- Python 3.10
- pip
- virtualenv
- cuda 12.1

## Installation

1. **Virtuelle Umgebung erstellen**

   Erstellen Sie eine neue virtuelle Umgebung namens `roboenv`:

   ```bash
   python3.10 -m venv roboenv_llama
   ```

2. **Virtuelle Umgebung aktivieren**

   Aktivieren Sie die virtuelle Umgebung:

   ```bash
   source roboenv/bin/activate
   ```

3. **Abhängigkeiten installieren**

   Installieren Sie die notwendigen Pakete mit pip:

   ```bash
   pip install -r requirements.txt
   ```

4**Anwendung starten**

   Starten Sie die Anwendung durch Ausführen der `setup_robodoc.py` Datei:

   ```bash
   python3.10 setup_robodoc.py
   ```


## Nutzung

Die Datei `inference.py` stelle die Schnittstelle dar, welche von Applikationen zum Abrufen verwendet wird, genauer stellt diese die Funktion chat_with_robodoc.

Die Erste Ausführung kann einige Zeit dauern, da das Modell zunächst heruntergeladen und gespeichert werden muss, anschließend wird es nur noch zum Bearbeiten von Anfragen in den VRAM geladen und anschließend unloaded.

Ein kleines Beispiel:


   ```bash
    from inference import chat_with_robodoc

    user_input = "The Patient has a mild fever."
    chat_history = []
    nodes_from_subgraph = "The Content: angioedema, disease, The Content: acquired angioedema, disease"
    image_captioning = "Male with brown hair." 

    # Invoke chat_with_robodoc with example data
    response_data = chat_with_robodoc(user_input, chat_history, nodes_from_subgraph, image_captioning)

    print("User Input:", response_data["user_input"])
    print("Model Response:", response_data["model_response"])
    print("Updated History:")
    for message in response_data["chat_history"]:
        print(f"{message['role'].capitalize()}: {message['content']}")
   ```

    
