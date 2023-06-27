import spacy
import scispacy
import tensorflow as tf
import itertools
import random
import os
import numpy as np 
from scispacy.linking import EntityLinker
from transformers import AutoModelForTokenClassification,pipeline, AutoModelForSequenceClassification,AutoTokenizer

#-------------------------------- NLP ----------------------------#

access_token='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'

fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP",use_auth_token=access_token)
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
fine_tuned_ner = pipeline("ner", model=fine_tuned_model, tokenizer=tokenizer,aggregation_strategy="simple")

age_tokenizer = AutoTokenizer.from_pretrained("padmajabfrl/Gender-Classification")
age_model = AutoModelForSequenceClassification.from_pretrained("padmajabfrl/Gender-Classification")
age_pipeline =pipeline("text-classification",model=age_model, tokenizer=age_tokenizer)


patient={"symptoms":[]}

def update_patient(model_output):
    prev_ent=''
    for g in model_output:
        entity =  g['entity_group']
        w = g['word']
        if(entity=='Age'):
            patient['age']=w
        elif(entity=='Sex'):
            patient['sex']=age_pipeline(w)[0]['label']
        elif(entity=='Sign_symptom'):
            if(prev_ent=='Sign_symptom'):
                prev =patient['symptoms'].pop()
                if(w.startswith('##')):
                    new=prev+w.replace('#','')
                else:
                    new =prev+' '+w
                patient['symptoms'].append(new)
            else:
                patient['symptoms'].append(w)
        prev_ent=entity


def reset_patient():
    patient = {"symptoms":[]}

def process_input(sentence):
    update_patient(fine_tuned_ner(sentence))
    return patient


#------------------------- PIPELINE_AND_PREDICTION ------------------#

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")

#loads rasan first predicting model
new_model = tf.keras.models.load_model(os.curdir+'/model_config')

#diseases dict
diseases=['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']
possible_symp = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic  patches', 'continuous sneezing', 'shivering', 'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue', 'vomiting', 'cough', 'chest pain', 'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain', 'yellowing of eyes', 'burning micturition', 'spotting  urination', 'passage of gases', 'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever', 'extra marital contacts', 'fatigue', 'weight loss', 'restlessness', 'lethargy', 'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger', 'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness', 'family history', 'mucoid sputum', 'headache', 'dizziness', 'loss of balance', 'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances', 'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium', 'dark urine', 'sweating', 'muscle pain', 'mild fever', 'swelled lymph nodes', 'malaise', 'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation', 'toxic look (typhos)', 'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'acute liver failure', 'swelling of stomach', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs', 'swollen blood vessels', 'prominent veins on calf', 'weight gain', 'cold hands and feets', 'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'abnormal menstruation', 'muscle weakness', 'anxiety', 'slurred speech', 'palpitations', 'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking', 'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads', 'scurring', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']


# Return a list of triplet tuples (UMLS concept ID, UMLS Canonical Name, Match score) for each entity that is in the given list of UMLS types
def filter_entities_by_types_group(umls_entities_list: list, type_group: list) -> list:
    return [(umls_entities_list[0][0].concept_id, umls_entities_list[0][0].canonical_name, umls_entities_list[0][1]) ]

def extract_umls_entities(kb_ents: list) -> list:
    # 1. T184 Sign or Symptom - return ALL and not only the first
    type_group_1 = ['T184']

    # 2. T037 Injury or Poisoning
    type_group_2 = ['T037']

    # 3. T004 Fungus | T005 Virus | T007 Bacterium | T033 Finding | T034 Laboratory or Test Result | T048 Mental or Behavioral Dysfunction
    type_group_3 = ['T004', 'T005', 'T007', 'T033', 'T034', 'T048']

    # 4. T047 Disease or Syndrome | T121 Pharmacologic Substance | T131 Hazardous or Poisonous Substance
    type_group_4 = ['T047', 'T121', 'T131']

    # Extract the UMLS Entities from each CUI value
    umls_entities_list = list(map(lambda ent: (linker.kb.cui_to_entity[ent[0]], ent[1]), kb_ents))

    type_1_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_1)
    type_2_matches_list = []
    type_3_matches_list = []
    type_4_matches_list = []

    if not type_1_matches_list:
        type_2_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_2)

        if not type_2_matches_list:
            type_3_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_3)

        if not type_3_matches_list:
            type_4_matches_list = filter_entities_by_types_group(umls_entities_list, type_group_4)

    secondary_matches_list = list(itertools.chain(type_2_matches_list, type_3_matches_list, type_4_matches_list))

    best_fit_secondary_value = 0
    best_fit_secondary_tuple = False

    # Keep only one secondary match at the most (the one with the best match value)
    for match in secondary_matches_list:
        if match[2] > best_fit_secondary_value:
            best_fit_secondary_tuple = match
            best_fit_secondary_value = match[2]

    if best_fit_secondary_tuple:
        type_1_matches_list.append(best_fit_secondary_tuple)

    return type_1_matches_list


def find_similarity(complaint_str: str)->list:
    if not isinstance(complaint_str, str):
        return [], []

    umls_entity_list = []
    complaint_str_parenthesis = '"{}"'.format(complaint_str)
    doc = nlp(complaint_str_parenthesis)
    for ent in doc.ents:
        if len(ent._.kb_ents) > 0 and len(ent._.kb_ents[0]) > 0:
            most_similar_list = extract_umls_entities(ent._.kb_ents)
            umls_entity_list+=most_similar_list

    return umls_entity_list

def symp_to_umls(symp_arr: list)-> list:
  umls=[]
  tuples =find_similarity('/ ' +" / ".join(symp_arr)+' /' )
  for tuple in tuples:
    umls.append(tuple[0])
  return umls

possible_symp_umls = sorted(symp_to_umls(possible_symp))


def pipeline(patient):
    symp = patient['symptoms']
    output_vector = [0] * len(possible_symp_umls)
    for i in range(len(symp)+2):
      random.shuffle(symp)
      umls_vector = symp_to_umls(symp)
      for word in umls_vector:
          if word in possible_symp_umls:
              index = possible_symp_umls.index(word)
              output_vector[index] = 1
    return output_vector

def decode_one_hot(pred):
  index = np.argmax(np.array(pred), axis=-1)
  print(diseases[index]+ "     Probability :"+str(pred[index]*100))
  return diseases[index]
  

def predict(nlp_output):
  return decode_one_hot(new_model.predict([pipeline(nlp_output)])[0])

#----------------------------#
def main():
    reset_patient()
    symptoms = process_input('I am 30 and a male, I have a headache, fever. I am also very tired.')
    print(symptoms)
    output=predict(symptoms)

if __name__ == '__main__':
    main()