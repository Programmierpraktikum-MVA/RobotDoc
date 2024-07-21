# Overview

![ablauf_anfrage](https://github.com/user-attachments/assets/e576f69c-1c46-4f1b-9145-0ba88681bcb9)



# Installation und Startanleitung

## 1. Klonen und Starten der Flask Anwendung auf dem GPU Server


1. **Repository Klonen und in aktuellen Branch wechseln**


   ```bash
   git clone https://github.com/Programmierpraktikum-MVA/RobotDoc.git
   git checkout main
   ```

2. **Virtuelle Umgebung einrichten**

   Erstellen und aktivieren Sie die virtuelle Umgebung:

   ```bash
	cd RobotDoc/03_application-team/
	python3.10 -m venv robodocvenv
	source robodocvenv/bin/activate
   ```

3. **Abhängigkeiten installieren**

   Installieren Sie die notwendigen Pakete mit pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **.env Datei erstellen**

   Kopieren der .env Datei in der 03_application-team/ Verzeichnis.


5. **Starten der Flask Anwendung**

	```bash
	nohup flask --app app.py run -h "öffentliche IP des Servers" -p 5005 &
	```

6. **Beenden der Flask Anwendung

	Zeigt flask Prozesse auf dem Server an
	```bash
	ps aux | grep flask
	```
	Herraussuchen aus der robodocvenv und "killen"
	```bash
	kill PID 
	```

7. **Starten und Bereitstellen von LLama und Llava**
    	
	https://github.com/Programmierpraktikum-MVA/RobotDoc/tree/KennyDev/03_application-team/modules/newmodel/Llama

	https://github.com/Programmierpraktikum-MVA/RobotDoc/tree/KennyDev/03_application-team/modules/newmodel/Llava
