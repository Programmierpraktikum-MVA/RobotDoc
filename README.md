# Overview

![ablauf_anfrage](https://github.com/user-attachments/assets/e576f69c-1c46-4f1b-9145-0ba88681bcb9)



# Installation und Startanleitung

## 1. Klonen und Starten der Flask Anwendung auf dem GPU Server


1. **Repository Klonen und in aktuellen Branch wechseln**


   ```bash
   git clone https://github.com/Programmierpraktikum-MVA/RobotDoc.git
   git checkout main
   ```

2. Backend Grundimage erstellen


   ```bash
	cd RobotDoc/backend/
	docker build -t robotdoc-backend-base -f Dockerfile.base .
   ```

3. Llama Grundimage erstellen

  

   ```bash
   cd RobotDoc/Llama/
	 docker build -t robotdoc-llama-base -f Dockerfile.base .
   ```

4. Llava Grundimage erstellen
   ```bash
   cd RobotDoc/Llava/
	 docker build -t robotdoc-llava-base -f Dockerfile.base .
   ```

5. Anwendung starten

	```bash
	 cd RobotDoc
   docker compose up
	```
