import tracemalloc
import spacy
import tensorflow as tf
import itertools
import random
import numpy as np 
from scispacy.linking import EntityLinker
from transformers import AutoModelForTokenClassification,pipeline, AutoModelForSequenceClassification,AutoTokenizer
#-------------------------------- NLP ----------------------------#
resetPatientBool = True

access_token='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'

fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP",use_auth_token=access_token)
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
fine_tuned_ner = pipeline("ner", model=fine_tuned_model, tokenizer=tokenizer,aggregation_strategy="simple")

age_tokenizer = AutoTokenizer.from_pretrained("padmajabfrl/Gender-Classification")
age_model = AutoModelForSequenceClassification.from_pretrained("padmajabfrl/Gender-Classification")
age_pipeline =pipeline("text-classification",model=age_model, tokenizer=age_tokenizer)


patient={"symptoms":[]}

def update_patient(nlp_output):
    """
    Updates patient (dict) with the new information provided by the NLP output

    Args:
    nlp_output (list) : Output list from the NLP where each word is assigned a category


    Returns:
    
    """
    prev_ent=''
    for g in nlp_output:
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
    """
    Reset the patient symptoms

    Args: None

    Returns: None

    """
    global patient 
    patient= {"symptoms":[]}

def process_input(sentence):
    """
    Update the patient informations given a sentence in natural language  
    Reset the patient symptoms if resetPatientBool is True (if the last disease prediction confidence was above a threshold )

    Args:
        sentence (str): Natural language string 
    
    Returns:
        patient (dict): Patient dictionary containing the symptoms(, age and gender)
    """
    if resetPatientBool : 
         reset_patient() 
    update_patient(fine_tuned_ner(sentence))
    return patient



#------------------------- PIPELINE_AND_PREDICTION ------------------#




def loadSciSpacy():
    """
    Load and return SciSpacy NLP for symptoms to UMLS classification

    Args: None

    Returns:
        nlp 
        linker
    """
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    return nlp, linker

nlp,linker = loadSciSpacy()

#Load the 2 prediction models
first_model = tf.keras.models.load_model('modules/model/model_config')
second_model = tf.keras.models.load_model('modules/model/model_config2')

#All possible diseases from the dataset
diseases=['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']

#All possible symptoms working with the first and the second model
symp_model_1 = ['itching', 'skin rash', 'nodal skin eruptions', 'dischromic  patches', 'continuous sneezing', 'shivering', 'chills', 'watering from eyes', 'stomach pain', 'acidity', 'ulcers on tongue', 'vomiting', 'cough', 'chest pain', 'yellowish skin', 'nausea', 'loss of appetite', 'abdominal pain', 'yellowing of eyes', 'burning micturition', 'spotting  urination', 'passage of gases', 'internal itching', 'indigestion', 'muscle wasting', 'patches in throat', 'high fever', 'extra marital contacts', 'fatigue', 'weight loss', 'restlessness', 'lethargy', 'irregular sugar level', 'blurred and distorted vision', 'obesity', 'excessive hunger', 'increased appetite', 'polyuria', 'sunken eyes', 'dehydration', 'diarrhoea', 'breathlessness', 'family history', 'mucoid sputum', 'headache', 'dizziness', 'loss of balance', 'lack of concentration', 'stiff neck', 'depression', 'irritability', 'visual disturbances', 'back pain', 'weakness in limbs', 'neck pain', 'weakness of one body side', 'altered sensorium', 'dark urine', 'sweating', 'muscle pain', 'mild fever', 'swelled lymph nodes', 'malaise', 'red spots over body', 'joint pain', 'pain behind the eyes', 'constipation', 'toxic look (typhos)', 'belly pain', 'yellow urine', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'acute liver failure', 'swelling of stomach', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'phlegm', 'blood in sputum', 'throat irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'loss of smell', 'fast heart rate', 'rusty sputum', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'cramps', 'bruising', 'swollen legs', 'swollen blood vessels', 'prominent veins on calf', 'weight gain', 'cold hands and feets', 'mood swings', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails', 'swollen extremeties', 'abnormal menstruation', 'muscle weakness', 'anxiety', 'slurred speech', 'palpitations', 'drying and tingling lips', 'knee pain', 'hip joint pain', 'swelling joints', 'painful walking', 'movement stiffness', 'spinning movements', 'unsteadiness', 'pus filled pimples', 'blackheads', 'scurring', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']
symp_model_2=[' abdominal_pain', ' abnormal_menstruation', ' acidity', ' acute_liver_failure', ' altered_sensorium', ' anxiety', ' back_pain', ' belly_pain', ' blackheads', ' bladder_discomfort', ' blister', ' blood_in_sputum', ' bloody_stool', ' blurred_and_distorted_vision', ' breathlessness', ' brittle_nails', ' bruising', ' burning_micturition', ' chest_pain', ' chills', ' cold_hands_and_feets', ' coma', ' congestion', ' constipation', ' feel_of_urine', ' sneezing', ' cough', ' cramps', ' dark_urine', ' dehydration', ' depression', ' diarrhoea', ' dischromic _patches', ' distention_of_abdomen', ' dizziness', ' drying_and_tingling_lips', ' enlarged_thyroid', ' excessive_hunger', ' extra_marital_contacts', ' family_history', ' fast_heart_rate', ' fatigue', ' fluid_overload', ' foul_smell_of urine', ' headache', ' high_fever', ' hip_joint_pain', ' history_of_alcohol_consumption', ' increased_appetite', ' indigestion', ' inflammatory_nails', ' internal_itching', ' irregular_sugar_level', ' irritability', ' irritation_in_anus', ' joint_pain', ' knee_pain', ' lack_of_concentration', ' lethargy', 'appetite_loss', 'balance_loss', 'smell_loss', ' malaise', ' mild_fever', ' mood_swings', ' movement_stiffness', ' mucoid_sputum', ' muscle_pain', ' muscle_wasting', ' muscle_weakness', ' nausea', ' neck_pain', ' nodal_skin_eruptions', ' obesity', 'behind_the_eyes_pain', 'during_bowel_movements_pain', ' anal_region_pain', ' walking_painful', ' palpitations', ' passage_of_gases', ' patches_in_throat', ' phlegm', ' polyuria', ' prominent_veins_on_calf', ' puffy_face_and_eyes', ' pus_filled_pimples', ' receiving_blood_transfusion', ' receiving_unsterile_injections', ' red_sore_around_nose', ' red_spots_over_body', ' redness_of_eyes', ' restlessness', ' runny_nose', ' rusty_sputum', ' scurring', ' shivering', ' silver_like_dusting', ' sinus_pressure', 'peeling_skin', ' skin_rash', ' slurred_speech', ' small_dents_in_nails', ' spinning_movements', ' spotting_ urination', ' stiff_neck', ' stomach_bleeding', ' stomach_pain', ' sunken_eyes', ' sweating', ' swelled_lymph_nodes', ' joints_swealling', ' swelling_of_stomach', ' swollen_blood_vessels', ' swollen_extremeties', ' swollen_legs', ' throat_irritation', ' toxic_look_(typhos)', ' ulcers_on_tongue', ' unsteadiness', ' visual_disturbances', ' vomiting', ' watering_from_eyes', ' weakness_in_limbs', ' weakness_of_one_body_side', 'gain_weight', 'loss_weight', ' yellow_crust_ooze', ' yellow_urine', ' yellowing_of_eyes', ' yellowish_skin', 'itching']

"""------CODE FROM ROBOTDOC 2022-------"""

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

"""------END CODE FROM ROBOTDOC 2022-------"""

def symp_to_umls(symptoms ,case):
    """
  Convert a symptoms list to a list with the corresponding UMLS codes

  Args:
    symptoms(list): list of symptoms to convert into UMLS codes
    case(int): 1 -> model1 - 2-> model2
  
  Returns:
    umls(list): List of UMLS codes
    """
    if case ==1 :   
      umls=[]
      tuples =find_similarity('/ ' +" / ".join(symptoms)+' /' )
      for tuple in tuples:
         umls.append(tuple[0])
      return umls
    
    elif case == 2:
     cleanSymp = [sub.strip().replace('_', ' ') for sub in symptoms]

     umls=[]
     for s in cleanSymp:
         subumls=[]
         tuples=find_similarity('/ '+s+' /')
         for tuple in tuples:
            subumls.append(tuple[0])
         umls.append(subumls)
     return umls
    

    
#UMLS code list, that will be used as input vector shape
umls_symp_model_1 = sorted(symp_to_umls(symp_model_1,1))
umls_symp_model_2 = symp_to_umls(symp_model_2,2)
"""print("UMLS S 1")
print(umls_symp_model_1)

print("UMLS S 2")
print(umls_symp_model_2)
"""
def generate_input_vector(patient,case):
    """
    Generate the input vector of the selected prediction model (1 or 2) given patient informations

    Args: 
        patient(dict): Patient dictionary containing the symptoms(, age and gender)
        case(int): 1 -> model1 - 2-> model2

    Returns:
        input_vector(list): vector where possible symptoms corresponding to the selected model are one-hot encoded
    """
    if(case==1):
        symp = patient['symptoms']
        input_vector = [0] * len(umls_symp_model_1)
        for i in range(len(symp)+1):
            random.shuffle(symp)
            umls_vector = symp_to_umls(symp,case)
            for word in umls_vector:
               if word in umls_symp_model_1:
                 index = umls_symp_model_1.index(word)
                 input_vector[index] = 1
        return input_vector
    elif(case==2):
        symp = patient['symptoms']
        input_vector = [0] * len(symp_model_2)
        for i in range(len(symp)+1):
            random.shuffle(symp)
            umls_vector = symp_to_umls(symp,case)
            for word in umls_vector:
               
                for id,u in enumerate(umls_symp_model_2):
                    if word[0] in u:
                     input_vector[id]=1 
                     print("OUIIII")
        return input_vector          

def decode_one_hot(model_prediction):
  """
  Extract from the prediction model output the most likely disease

  Args:
    model_prediction(list) : List containing for each possible diseases a confidence probability

  Returns:
    disease_prediction ((str,float)) : Tuple containing the most likely disease and its confidence probability
    
  
  """
  index = np.argmax(np.array(model_prediction), axis=-1)
  disease_prediction = (diseases[index],model_prediction[index])
  return disease_prediction
  

def decode_symp(input_vect):
    """From an input vector of the 2nd model output the symptoms that it encodes

        Args:
            input_vect(list): input vector of the 2nd model
        
        Returns:
            symptoms(list): List of symtoms that are encoded in the input vector

    
    """
    symptoms = np.array(symp_model_2)[np.array(input_vect)==1].tolist()
    return symptoms

def predict(patient, threshold = 0.25):
    """
    Returns the prediction of the model if the confidence probability is above the threshold 

    Args:
        patient(dict): Patient dictionary containing the symptoms(, age and gender)
        threshold(int) (default: 0.25): threshold to reach to return the prediction 
    
    Returns:
        prediction((str,int)): Tuple containing the predicted disease or an error message and the actual confidence probability
    
    """
    global resetPatientBool 
    input_vector = generate_input_vector(patient,2)
    patient_symp = decode_symp(input_vector)
    model1_prediction = decode_one_hot(first_model.predict([generate_input_vector(patient,1)])[0])
    model2_prediction = decode_one_hot(second_model.predict([input_vector])[0])   
    print("PATIENT SYMP")
    print( patient_symp) 
    most_likely_prediction = model1_prediction if model1_prediction[1]>model2_prediction[1] else model2_prediction
    if most_likely_prediction[1]<threshold:
        resetPatientBool = False
        return ("We can't provide a good diagnostic, please provide us more informations",patient_symp,most_likely_prediction[1])
    else:
        resetPatientBool = True
        return (most_likely_prediction[0],patient_symp,most_likely_prediction[1])
     
#----------------------------#
def main():
    reset_patient()

if __name__ == '__main__':
    main()

    