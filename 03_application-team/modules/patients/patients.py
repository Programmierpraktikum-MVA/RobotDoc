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
    try:
        cleanOutput = getSymptoms(textToconvert, 'mlteam')
    except:
        cleanOutput = "Error"
    print(cleanOutput)
    return render_template("home.html", user=str(current_user.id), output=cleanOutput, initialText=textToconvert)


# prediction for user
@patients.route("/assignTokens/<int:id>", methods=["POST"])
@login_required
def assignTokens(id):
    textToconvert = request.form.get("textToConvert")
    try:
        symptoms = getSymptoms(textToconvert, 'mlteam')
        patientData[id-1]["symptoms"].append(symptoms['symptoms']) # assign symptoms to patient
        cleanOutput = getDiagnosis(symptoms, 'mlteam') 
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
