# RobotDoc 

Automatische Diagnose – a) Identifikation von Mustern, b) automatische Anerkennung von Krankheiten und c) Vorschlag von Behandlung. [^1]

## Set-up

0. create and select virtual environment (with Python 3.9.6 or lower)
1. choose application: ```cd 03_application-team```
2. install requirements: ```pip install -r requirements.txt```
3. start application: ```flask --app app.py run```
4. open application: http://127.0.0.1:5000

### Debug

- BASIC: restart ```venv```, terminal, and IDE
- ERROR: ```Could not build wheels for nmslib, which is required to install pyproject.toml-based projects``` ⇒ use older version of Python (tested up to ```3.9.6```)
- ERROR: ```Getting requirements to build wheel did not run successfully.``` (e.g. ```psycopg2```) ⇒ ```pip install psycopg2-binary``` (```pip install``` all other dependencies manually, ```en-core-sci-sm``` via ```pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz```)
- ERROR: ```AutoModeLForTokenClassification requires the PyTorch library but it was not found in your environment.``` ⇒ restart ```venv``` and terminal then select correct ```venv```

## Demo

https://github.com/Programmierpraktikum-MVA/RobotDoc/assets/79874249/0f912dda-ca5f-4f4d-b21f-f1fd5ed1a099
> state: 22.05.

[^1]: https://isis.tu-berlin.de/course/view.php?id=33313#section-
