
# LLama3 Installation und Startanleitung

## Voraussetzungen
- Python 3.8 oder höher
- pip
- virtualenv
- cuda

## Installation

1. **Virtuelle Umgebung erstellen**

   Erstellen Sie eine neue virtuelle Umgebung namens `roboenv`:

   ```bash
   virtualenv roboenv
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

4. **Hugging Face Token konfigurieren**

   In der `config.json` ersetzen Sie `TOKEN_HERE` in der mit Ihrem tatsächlichen Hugging Face Token.

   ```json
   {"HF_TOKEN":"TOKEN_HERE"}
   ```
   

5. **Anwendung starten**

   Starten Sie die Anwendung durch Ausführen der `startup.py` Datei:

   ```bash
   python setup.py
   ```
      

## Nutzung

Nachdem Sie LLama3 installiert und die `startup.py` ausgeführt haben, sollte die Anwendung laufen und bereit für die Nutzung sein, die Datei `inference.py` stelle dabei die Schnittstelle dar, welche von Applikationen zum Abrufen verwendet wird.

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

    # Display the results
    print("User Input:", response_data["user_input"])
    print("Model Response:", response_data["model_response"])
    print("Updated History:")
    for message in response_data["chat_history"]:
        print(f"{message['role'].capitalize()}: {message['content']}")
   ```

    
