
import networkx as nx
import pandas as pd
import os
import openpyxl
import ast
import csv
#from model.py import symp_model_1, symp_model_2


# importing the csv file
current_directory = os.getcwd()

#DENEME, PATH PROBLEM ÇIKARIYOR CHATGPT YARDIMI:
# Specify the subdirectory (util) and the filename (dataset.csv)
subdirectory = "util"
filename = "dataset.csv"

# Create the full file path by concatenating the subdirectory and filename
file_path = os.path.join(current_directory, subdirectory, filename)


#file_path = current_directory + '\\MimicIV_Version_18.06.23 (1).xlxs'
#DENEME : BURANIN ALTI BENİM KULLANDIĞIM
#file_path = current_directory + '\\dataset.csv'

#mimic_data = pd.read_excel( 'MimicIV_Version_18.06.23 (1).xlsx' , nrows= 1000)
#mimic_data = symp_model_1
#DENEME : VE BURANIN ALTI
mimic_data = pd.read_csv(file_path, nrows=1000)
#mimic_data = pd.read_csv( 'dataset.csv' , nrows= 1000)

"""""""""
# returns a tuple list of input nodes
def extract_input_nodes_as_tuples(df):


    # (code, description)
    uml_tuples = []
    uml_descriptions = []
    uml_codes = []


    for index, row in df.iterrows():

        #a = df.at[index,  'umls_canonical_name_list']
        a = df.at[index, 'Disease']

        b = df.at[index,  'Symptom_1']
        symptoms = df.iloc[:, 1:]

        #a_list = eval(a)
        #b_list = eval(b)
        a_list = list(csv.reader([a]))[0]  # Parse 'a' as CSV and convert it to a list
        b_list = list(csv.reader([b]))[0]  # Parse 'b' as CSV and convert it to a list

        for x in range(len(a_list)):
            if a_list[x] not in uml_descriptions and b_list[x] not in uml_codes:
                uml_descriptions.append(a_list[x])
                uml_codes.append(b_list[x])

                uml_tuples.append((b_list[x], a_list[x]))

    return uml_tuples

# returns a tuple-list of output nodes
def extract_output_nodes_as_tuples(df):

    uml_tuples = []
    uml_description = []
    uml_code =[]

    df['Diagnose_Combined'] = list(zip(df['icd_code'], df['icd_title']))

    for index, row in df.iterrows():

        tuple = df.at[index,  'Diagnose_Combined']
        

        if tuple[0] not in uml_code and tuple[1] not in uml_description:

            uml_code.append(tuple[0])
            uml_description.append(tuple[1])

            uml_tuples.append(tuple)
    
    return uml_tuples

def define_patient_tuples(df):

    patient_ids= []

    for index, row in df.iterrows():

        subject_id = df.at[index,  'subject_id']

        patient_ids.append(subject_id)
    
    return patient_ids
    """""""""

# Returns a tuple list of input nodes
"""""""""
def extract_input_nodes_as_tuples(df):
    uml_tuples = []
    uml_descriptions = []
    uml_codes = []

    for index, row in df.iterrows():
        a_list = [df.at[index, f'Symptom_{i}'] for i in range(1, 18)]  # Update column names accordingly

        for x in range(len(a_list)):
            if a_list[x] not in uml_descriptions:
                uml_descriptions.append(a_list[x])

                # You can use a different method to generate codes for symptoms
                uml_codes.append(f'Symptom_{x+1}')

                uml_tuples.append((f'Symptom_{x+1}', a_list[x]))

    return uml_tuples
"""""""""
def extract_input_nodes_as_tuples(df):
    uml_tuples = []

    for index, row in df.iterrows():
        diagnosis = row['Disease']
        symptoms = [row[f'Symptom_{i}'] for i in range(1, 18)]

        for symptom in symptoms:
            # Check if the symptom is not empty before adding it
            if symptom:
                uml_tuples.append((symptom, diagnosis))

    return uml_tuples
# Returns a tuple-list of output nodes
def extract_output_nodes_as_tuples(df):
    uml_tuples = []
    uml_description = []
    uml_code = []

    for index, row in df.iterrows():
        diagnosis = df.at[index, 'Disease']  # Update column name accordingly
        for i in range(1, 18):  # Assuming you have 17 symptom columns
            symptom = df.at[index, f'Symptom_{i}']
            tuple = (f'Symptom_{i}', symptom)  # Update accordingly

            if tuple[0] not in uml_code and tuple[1] not in uml_description:
                uml_code.append(tuple[0])
                uml_description.append(tuple[1])

                uml_tuples.append(tuple)

        # You can also add the diagnosis as a node
        uml_code.append(diagnosis)
        uml_description.append(diagnosis)
        uml_tuples.append((diagnosis, diagnosis))

    return uml_tuples



