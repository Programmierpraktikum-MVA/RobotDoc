from flask import Flask, Response, url_for, request, session, abort, render_template, redirect, jsonify, Blueprint
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.db_model import *
from util.funcs import *
import modules.model.model as ml

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
    except Exception as e:
        cleanOutput = "Error"
    print(cleanOutput)
    
    return render_template("home.html", user=str(current_user.id), prediction=cleanOutput[0], symptoms=cleanOutput[1],confidence=round(cleanOutput[2]*100,3),initialText=textToconvert)

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
    try:
        if nlp == 1:
            symptoms = getSymptoms(textToconvert, NLP.HF)
            patientData[id-1]["symptoms"].append(symptoms)
            cleanOutput = 'This model does not support prediction!'
        if nlp == 2:
            symptoms = getSymptoms(textToconvert, NLP.MLTEAM)
            patientData[id-1]["symptoms"].append(symptoms['symptoms']) # assign symptoms to patient
            cleanOutput = getDiagnosis(symptoms, PM.MLTEAM) 
    except:
        cleanOutput = "Error"
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
    return render_template("patientSpec.html", patientData=patientData[id-1], prediction=cleanOutput, initialText=textToconvert)


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