from flask import Flask, Response, app, url_for, request, session, abort, render_template, redirect, jsonify, Blueprint, current_app
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from util.db_model import *
from util.funcs import *
#import modules.model.model as ml
import numpy as np
from modules.newmodel import subgraphExtractor, llm

from util.cache_config import cache
#Imports for Graph-Drawing

import networkx as nx
import matplotlib.pyplot as plt


import base64

from transformers import AutoModelForTokenClassification,pipeline, AutoModelForSequenceClassification,AutoTokenizer
from modules.KnowledgeExtraction import subgraph_builder





patients = Blueprint("patients", __name__)


fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
finetuned_ner = pipeline("ner", model=fine_tuned_model, tokenizer=tokenizer,aggregation_strategy="simple")


@cache.memoize(timeout=300)
def getAllPatients():
    with current_app.app_context():
        data = Patients.query.all()
        return data
    
def patients_to_dict(patients):
    return {patient.id: patient for patient in patients}

def getPatient(id):
    with current_app.app_context():
        patient = Patients.query.get(id)
        return patient
    




@patients.route("/patients")
@login_required
def patientsView():
    patientData = getAllPatients()
    return render_template("patients.html", patients=patientData)


#KNOWLEDGE GRAPH Dummy-Page
@patients.route("/know")
@login_required
def knowView():
    return render_template("know.html")


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

     # Add nodes (symptoms) and edges (relationships)
    symptoms = cleanOutput[1]
    disease = cleanOutput[0]
    print(symptoms)
    print(type(symptoms))
    # Drawing a Graph
    # Create a new graph

    G = nx.DiGraph()

    # Add nodes (symptoms) and edges (relationships)
    # All possible diseases from the dataset
    all_diseases = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy',
                    'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
                    'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
                    'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C',
                    'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism',
                    'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)',
                    'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid',
                    'Urinary tract infection', 'Varicose veins', 'hepatitis A']
    # All possible symptoms working with the first and the second model
    all_symptoms = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic  patches', 'continuous sneezing',
                    'shivering', 'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue',
                    'vomiting', 'cough', 'chest pain', 'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain',
                    'yellowing of eyes', 'burning micturition', 'spotting  urination', 'passage of gases',
                    'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever',
                    'extra marital contacts', 'fatigue', 'weight loss', 'restlessness', 'lethargy',
                    'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger',
                    'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness',
                    'family history', 'mucoid sputum', 'headache', 'dizziness', 'loss of balance',
                    'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances',
                    'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium',
                    'dark urine', 'sweating', 'muscle pain', 'mild fever', 'swelled lymph nodes', 'malaise',
                    'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation', 'toxic look (typhos)',
                    'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections',
                    'coma', 'stomach bleeding', 'acute liver failure', 'swelling of stomach', 'distention of abdomen',
                    'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum',
                    'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion',
                    'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements',
                    'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs',
                    'swollen blood vessels', 'prominent veins on calf', 'weight gain', 'cold hands and feets',
                    'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties',
                    'abnormal menstruation', 'muscle weakness', 'anxiety', 'slurred speech', 'palpitations',
                    'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking',
                    'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads',
                    'scurring', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'skin peeling',
                    'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister',
                    'red sore around nose', 'yellow crust ooze']

    symptoms = [symptom.strip() for symptom in symptoms]
    if not isinstance(symptoms, list):
        symptoms = [symptoms]

    for possible_disease in all_diseases:
        G.add_node(possible_disease)


    # for symptom in all_symptoms:
    #    G.add_node(symptom)

    for symptom in symptoms:
        # G.add_node(symptom)
        G.add_edge(symptom, disease, arrows=True, arrowstyle="-|>", arrowsize=30)

    # create empty list for node colors
    node_color = []

    # for each node in the graph
    for node in G.nodes(data=True):
        if node[0] in all_symptoms:
            node_color.append("#69A8F5")
        else:
            node_color.append("#F79A28")

    #Drawing the graph
    matplotlib.use('agg')

    plt.figure(figsize=(10, 10))
    outer_nodes = set(G) - set(symptoms)
    pos = nx.circular_layout(G.subgraph(outer_nodes), scale=3)  #
    pos2 = nx.circular_layout(G.subgraph(symptoms), center=[0, 0], scale=0.5)
    pos_fin = {**pos, **pos2}
    pos_fin

    # nx.draw_shell(G,k=20, with_labels=True, font_size=6, node_size=50, node_color=node_color , font_color='black')
    nx.draw(G, pos_fin, with_labels=True, font_size=8, node_size=500, node_color=node_color, font_color='black')
    # nx.draw(G, pos, with_labels=True)
    plt.title(f"Knowledge Graph for {disease}")

    # plt.show()
    # Save the graph as an image
    plt.savefig(
        "static/img/graph_visualization.png")

    return render_template("home.html", user=str(current_user.id), prediction=cleanOutput[0], symptoms=[s.strip().replace('_',' ') for s  in cleanOutput[1]]
    ,confidence=round(cleanOutput[2]*100,3),initialText=textToconvert)


@patients.route("/resetSymptoms", methods=["POST"])
@login_required
def resetSymptoms():
    # ml.reset_patient()
    return render_template("home.html", user=str(current_user.id))

# prediction for user
@patients.route("/assignTokens/<int:id>/<int:nlp>", methods=["POST"])
@login_required
def assignTokens(id,nlp):
    patientData = getAllPatients()

    textToconvert = request.form.get("textToConvert")
    threshold = int(request.form.get("threshold"))/100
    try:
        if nlp == 1:
            symptoms = getSymptoms(textToconvert, NLP.HF)
            patientData[id]["symptoms"].append(symptoms)
            cleanOutput = 'This model does not support prediction!'
        if nlp == 2:
            symptoms = getSymptoms(textToconvert, NLP.MLTEAM)
            # ml.reset_patient()
            # patient_symptoms = add_patient_symptoms(id,symptoms['symptoms'])
            # print(patient_symptoms['symptoms'])
             # assign symptoms to patient
            cleanOutput = [None,None]
            patientData[id]["symptoms"] = [s.strip().replace('_',' ') for s  in cleanOutput[1]]
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
    # Add nodes (symptoms) and edges (relationships)
    symptoms = cleanOutput[1]
    disease = cleanOutput[0]
    print(symptoms)
    print(type(symptoms))

    G = nx.DiGraph()

    # Add nodes (symptoms) and edges (relationships)
    # All possible diseases from the dataset
    all_diseases = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy',
                    'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis',
                    'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction',
                    'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C',
                    'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism',
                    'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)',
                    'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid',
                    'Urinary tract infection', 'Varicose veins', 'hepatitis A']
    # All possible symptoms working with the first and the second model
    all_symptoms = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic  patches', 'continuous sneezing',
                    'shivering', 'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue',
                    'vomiting', 'cough', 'chest pain', 'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain',
                    'yellowing of eyes', 'burning micturition', 'spotting  urination', 'passage of gases',
                    'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever',
                    'extra marital contacts', 'fatigue', 'weight loss', 'restlessness', 'lethargy',
                    'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger',
                    'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness',
                    'family history', 'mucoid sputum', 'headache', 'dizziness', 'loss of balance',
                    'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances',
                    'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium',
                    'dark urine', 'sweating', 'muscle pain', 'mild fever', 'swelled lymph nodes', 'malaise',
                    'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation', 'toxic look (typhos)',
                    'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections',
                    'coma', 'stomach bleeding', 'acute liver failure', 'swelling of stomach', 'distention of abdomen',
                    'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum',
                    'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion',
                    'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements',
                    'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs',
                    'swollen blood vessels', 'prominent veins on calf', 'weight gain', 'cold hands and feets',
                    'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties',
                    'abnormal menstruation', 'muscle weakness', 'anxiety', 'slurred speech', 'palpitations',
                    'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking',
                    'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads',
                    'scurring', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'skin peeling',
                    'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister',
                    'red sore around nose', 'yellow crust ooze']

    symptoms = [symptom.strip() for symptom in symptoms]
    if not isinstance(symptoms, list):
        symptoms = [symptoms]

    for possible_disease in all_diseases:
        G.add_node(possible_disease)

    # for symptom in all_symptoms:
    #    G.add_node(symptom)

    for symptom in symptoms:
        # G.add_node(symptom)
        G.add_edge(symptom, disease, arrows=True, arrowstyle="-|>", arrowsize=30)

    # create empty list for node colors
    node_color = []

    # for each node in the graph
    for node in G.nodes(data=True):
        if node[0] in all_symptoms:
            node_color.append("#69A8F5")
        else:
            node_color.append("#F79A28")

    matplotlib.use('agg')

    plt.figure(figsize=(10, 10))
    outer_nodes = set(G) - set(symptoms)
    pos = nx.circular_layout(G.subgraph(outer_nodes), scale=3)  #
    pos2 = nx.circular_layout(G.subgraph(symptoms), center=[0, 0], scale=0.5)
    pos_fin = {**pos, **pos2}
    pos_fin

    # nx.draw_shell(G,k=20, with_labels=True, font_size=6, node_size=50, node_color=node_color , font_color='black')
    nx.draw(G, pos_fin, with_labels=True, font_size=8, node_size=500, node_color=node_color, font_color='black')
    # nx.draw(G, pos, with_labels=True)
    plt.title(f"Knowledge Graph for {disease}")

    plt.savefig(
        "static/img/graph_visualization_patient.png")
    patientData = getAllPatients()


    return render_template("patientSpec.html", patientData=patientData[id], prediction=cleanOutput[0], initialText=textToconvert, confidence =round(cleanOutput[2]*100,3) )


@patients.route("/patients/<int:id>")
@login_required
def patients_route(id):
    patientData = getPatient(id).to_dict()

    imagesOfPatient = Image.query.filter_by(patient_id=id).all()

    # Convert the bytea data to base64
    image_data = []
    for image in imagesOfPatient:
        encoded_image = base64.b64encode(image.file).decode('utf-8')  # Assuming `image.file` contains the binary data
        image_data.append({
            "id": image.id,
            "url": f"data:image/jpeg;base64,{encoded_image}"  # Adjust the MIME type if necessary
        })

    print("You pressed on: " + str(id))
    return render_template("patientSpec.html", patientData=patientData, imagesOfPatient=image_data)


@patients.route("/patients/<int:patientID>/symptoms/<int:symptomIndex>")
@login_required
def deleteSymptoms(symptomIndex, patientID):
     with current_app.app_context():
        # Save the changes to the database
        try:
             # Get the patient from the database
            patient = Patients.query.get(patientID)
            # Convert the symptoms from TEXT[] to a list
            symptoms_list = list(patient.symptoms)

            # Check if symptomIndex is valid
            if symptomIndex < 0 or symptomIndex >= len(symptoms_list):
                return "Invalid symptom ID.", 400

            # Remove the symptom at the given index
            symptoms_list.pop(symptomIndex)

            # Update the patient's symptoms
            patient.symptoms = symptoms_list

            # Save the changes to the database
            db.session.commit()
            cache.delete_memoized(getAllPatients)
        except Exception as e:
            db.session.rollback()
            return str(e), 500

        # Redirect the user to the patient's page
        return redirect("/patients/" + str(patientID))




@patients.route("/editPage/<int:id>")
@login_required
def editPage(id):
    data = getAllPatients()
    patientData = patients_to_dict(data)
    imagesOfPatient = Image.query.filter_by(patient_id=id).all()

    # Convert the bytea data to base64
    image_data = []
    for image in imagesOfPatient:
        encoded_image = base64.b64encode(image.file).decode('utf-8')  # Assuming `image.file` contains the binary data
        image_data.append({
            "id": image.id,
            "url": f"data:image/jpeg;base64,{encoded_image}"  # Adjust the MIME type if necessary
        })

    print("You pressed on: " + str(id))
    return render_template("edit-patient.html", patientData=patientData[id], imagesOfPatient=image_data)

@patients.route("/editPatient/<int:id>", methods=["POST"])
def edit_patient(id):
    with current_app.app_context():


        # Save the changes to the database
        try:
             # Get the patient from the database
            patient = Patients.query.get(id)


            # Update the patient's data with the form data
            patient.name = str(request.form['name']).strip()
            patient.age = int(request.form['age'])
            patient.weight =float(request.form['weight'])
            patient.sex = request.form['sex']
            patient.symptoms = [symptom.strip() for symptom in request.form['symptoms'].split(',')]
            
            db.session.commit()
            cache.delete_memoized(getAllPatients)
        except Exception as e:
            db.session.rollback()
            return str(e), 500
    
    imagesOfPatient = Image.query.filter_by(patient_id=id).all()

    # Convert the bytea data to base64
    image_data = []
    for image in imagesOfPatient:
        encoded_image = base64.b64encode(image.file).decode('utf-8')  # Assuming `image.file` contains the binary data
        image_data.append({
            "id": image.id,
            "url": f"data:image/jpeg;base64,{encoded_image}"  # Adjust the MIME type if necessary
        })

    # Redirect the user to the patient's page
    return render_template("patientSpec.html", patientData=getPatient(id).to_dict(), imagesOfPatient=image_data)

@patients.route("/deleteImage/<int:image_id>", methods=["POST"])
@login_required
def deleteImage(image_id):
    image = Image.query.get_or_404(image_id)
    db.session.delete(image)
    db.session.commit()
    return redirect(request.referrer)  # Redirect back to the previous page

@patients.route("/sendMessage/<int:id>", methods=["POST"])
def sendMessage(id):
    data = request.get_json()
    message = data['message']

    print(id)
    if(data['updateSymptoms']):
        symps = subgraphExtractor.symptomNER(message)
        if len(symps) > 0:
            print("symp length works")
            with current_app.app_context():
                try:
                    patient = Patients.query.get(id)
                    currentSymptoms = patient.symptoms
                    uniqueSymptoms = []
                    # Iterate over new symptoms
                    for s in symps:
                        # Check if the symptom is not a substring in any of the current symptoms
                        if not any(s in cs for cs in currentSymptoms):
                            uniqueSymptoms.append(s)
                    print(uniqueSymptoms)
                    if len(uniqueSymptoms) > 0:
                        return jsonify({"reply": uniqueSymptoms, "type": "symptoms"})
                except Exception as e:
                    return jsonify({"reply": e, "type": "message"}), 500

    reply = subgraphExtractor.processMessage(message)
    return jsonify({"reply": reply, "type": "message"})
    



@patients.route("/updateSymptoms/<int:id>", methods=["POST"])
def updateSymptoms(id):
    with current_app.app_context():
        try:
            data = request.get_json()
            symp = data['symptoms']
            patient = Patients.query.get(id)
            patient.symptoms = [symptom.strip() for symptom in symp.split(',')]
            db.session.commit()
            cache.delete_memoized(getAllPatients)
            return jsonify({"success": True})
        except Exception as e:
            db.session.rollback()
            return str(e), 500