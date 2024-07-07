import os
import pandas as pd

# importing the csv file

def get_data(number_of_rows):
    current_directory = os.getcwd()

    # DENEME, PATH PROBLEM ÇIKARIYOR CHATGPT YARDIMI:
    # Specify the subdirectory (util) and the filename (dataset.csv)
    subdirectory = "util"
    filename = "dataset.csv"

    # Create the full file path by concatenating the subdirectory and filename
    file_path = os.path.join(current_directory, subdirectory, filename)

    #file_path = current_directory + '\\MimicIV_Version_18.06.23 (1).xlxs'#
    #file_path = current_directory + '\\dataset.csv'
    #print(file_path)

    #mimic_data = pd.read_excel('MimicIV_Version_18.06.23 (1).xlsx', nrows=number_of_rows)
    mimic_data = pd.read_csv(file_path, nrows=number_of_rows)

    return mimic_data
