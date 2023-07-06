# RobotDoc 

Automatische Diagnose – a) Identifikation von Mustern, b) automatische Anerkennung von Krankheiten und c) Vorschlag von Behandlung. [^1]

## Set-up

1. choose application: ```cd 03_application-team```
2. install requirements: ```pip install -r requirements.txt```
3. start application: ```flask --app app.py --debug run```
4. open application: http://127.0.0.1:5000

### Debug

- basic
    - create a new ```venv```, install all dependencies, and start / select it
    - restart ```venv```, terminal, and IDE
- specific
    - error: ```Getting requirements to build wheel did not run successfully.``` (e.g. ```psycopg2```) => ```pip install psycopg2-binary``` 
        - ```pip install``` all other dependencies
        - for ```en-core-sci-sm```: ```pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz```
    - error: ```AutoModeLForTokenClassification requires the PyTorch library but it was not found in your environment.``` => restart (```venv```, terminal) and select correct ```venv```
    - run ```model.py``` once to download and cache datasets (will take a long time) - or get dataset from a peer (```Useers/[username]/.scispacy/datasets```)

## Demo

https://github.com/Programmierpraktikum-MVA/RobotDoc/assets/79874249/0f912dda-ca5f-4f4d-b21f-f1fd5ed1a099
> state: 22.05.

[^1]: https://isis.tu-berlin.de/course/view.php?id=33313#section-
