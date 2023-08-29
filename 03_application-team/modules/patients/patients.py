from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify, Blueprint
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.db_model import *
from util.funcs import *
import modules.model.model as ml
import numpy as np

patients = Blueprint("patients", __name__)


@patients.route("/patients")
@login_required
def patientsView():
    return render_template("patients.html", patients=patientData)

# general prediction without user ("Home")
@patients.route("/sendInput", methods=["POST"])
@login_required
def convertText():
    textToconvert = request.form.get("textToConvert")
    threshold = int(request.form.get("threshold"))/100
    try:
        symptoms = getSymptoms(textToconvert, NLP.MLTEAM)
        cleanOutput =getDiagnosis(symptoms, PM.MLTEAM,threshold)
    except:
        cleanOutput = "Error"
    print(cleanOutput)
    
    return render_template("home.html", user=str(current_user.id), prediction=cleanOutput[0], symptoms=[s.strip().replace('_',' ') for s  in cleanOutput[1]]
    ,confidence=round(cleanOutput[2]*100,3),initialText=textToconvert)

@patients.route("/resetSymptoms", methods=["POST"])
@login_required
def resetSymptoms():
    ml.reset_patient()
    return render_template("home.html", user=str(current_user.id))

# prediction for user
@patients.route("/assignTokens/<int:id>/<int:nlp>", methods=["POST"])
@login_required
def assignTokens(id,nlp):
    textToconvert = request.form.get("textToConvert")
    threshold = int(request.form.get("threshold"))/100
    try:
        if nlp == 1:
            symptoms = getSymptoms(textToconvert, NLP.HF)
            patientData[id-1]["symptoms"].append(symptoms)
            cleanOutput = 'This model does not support prediction!'
        if nlp == 2:
            symptoms = getSymptoms(textToconvert, NLP.MLTEAM)
            ml.reset_patient()
            patient_symptoms = add_patient_symptoms(id,symptoms['symptoms'])
            print(patient_symptoms['symptoms'])
             # assign symptoms to patient
            cleanOutput = getDiagnosis(patient_symptoms, PM.MLTEAM,threshold) 
            patientData[id-1]["symptoms"] = [s.strip().replace('_',' ') for s  in cleanOutput[1]]
            print(cleanOutput[1])
            
    except Exception as e :
        print(e)
        cleanOutput = ("Error",'','')
    """ old api 
    try:
        output = query({
            "inputs": textToconvert
        })
        parsedOutput = parseString(output)
    except:
        parsedOutput = "Error"
    if "Sign_symptom" in parsedOutput:
        print(parsedOutput["Sign_symptom"])
        patientData[id-1]["symptoms"].append(parsedOutput["Sign_symptom"])
    """
    print(cleanOutput)
    return render_template("patientSpec.html", patientData=patientData[id-1], prediction=cleanOutput[0], initialText=textToconvert, confidence =round(cleanOutput[2]*100,3) )


@patients.route("/patients/<int:id>")
@login_required
def patients_route(id):
    print("You pressed on: " + str(id))
    return render_template("patientSpec.html", patientData=patientData[id-1])


@patients.route("/patients/<int:patientID>/symptoms/<int:symptomID>")
@login_required
def deleteSymptoms(symptomID, patientID):
    print("PatientID: " + str(patientID))
    print("SymptomID: " + str(symptomID))
    symptom = next((patient["symptoms"].pop(symptomID) for patient in patientData if patient["id"] == patientID), None)
    print(patientData)
    return redirect("/patients/" + str(patientID))