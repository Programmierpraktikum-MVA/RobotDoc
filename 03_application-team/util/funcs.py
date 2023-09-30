import requests
from util.exceptions import *
import modules.model.model as ml
from util.db_model import *
from enum import Enum
#DENEME IMPORTLAR
from util.knowledge_graph import *
import matplotlib

# huggingface api (reference: https://huggingface.co/d4data/biomedical-ner-all)
API_URL = "https://api-inference.huggingface.co/models/d4data/biomedical-ner-all"
headers = {"Authorization": "Bearer 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"}

# available natural language models
class NLP(Enum):
    HF = 1
    MLTEAM = 2

# available prediction models
class PM(Enum):
    MLTEAM = 1

#DENEME: GRAPH ÇİZİMİ BİTMEDİ!!

def drawGraph(input):

    return 0

def getSymptoms(input, model):
    if model == NLP.HF:   
        payload = {
            "inputs": input
            } # get correct format
        response = convertString(requests.post(API_URL, headers=headers, json=payload).json) # api call
        if "Sign_symptom" in response: return response["Sign_symptom"]
        
    if model == NLP.MLTEAM:
        return ml.process_input(input) # get symptoms (NLP)
    else: raise InvalidModelError

def getDiagnosis(symptoms, model,threshold):
    if model == PM.MLTEAM:
        # NOTE: symptoms need to be in the format: {"symptoms":[]}
        return ml.predict(symptoms,threshold) # get diagnosis (prediction)
    else: raise InvalidModelError

def convertString(data):
    output = []
    for d in data:
        score_pct = round(d['score'] * 100, 2)
        line = f"{d['entity_group']}: {d['word']} (score: {score_pct}%, start: {d['start']}, end: {d['end']})"
        output.append(line)
    output.append("")
    return output


def parseString(data):
    # Create a dictionary to hold the arrays
    entity_groups = {}
    # Iterate over each item in the list
    for item in data:
        entity_group = item['entity_group']
        word = item['word']
        # If the entity group already exists, append the word to the existing array
        if entity_group in entity_groups:
            entity_groups[entity_group].append(word)
        # Otherwise, create a new array for the entity group
        else:
            entity_groups[entity_group] = [word]
    return entity_groups


def validate_username(username):
    """
    Username has to be alphanumeric and at least on character long.
    Raises InvalidUsername Exception.
    :param username: to validate
    :return: None if valid
    """
    if not str(username).isalnum():
        raise InvalidUsernameError
    return


def validate_password(password):
    """
    Password has to be at least 8 characters long.
    Raises InvalidPassword Exception
    :param password: to validate
    :return: None if valid
    """
    if len(password) < 8:
        raise InvalidPasswordError
    return

def add_patient_symptoms(id,symp):
    data = patientData[id-1]["symptoms"]
    for s in symp:
        if s not in data:
            data.append(s)
    patientData[id-1]["symptoms"] = data
    return  patientData[id-1]
    
    
    