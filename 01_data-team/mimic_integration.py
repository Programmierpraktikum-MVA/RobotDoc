import pandas as pd

pd.set_option('display.width', 1000)

#Load relevant datafarmes
ftd = pd.read_csv('Mimic IV/final_triage_data.csv') #Informatio about Symptoms - Disease aswell as Patient - Disease
kg = pd.read_csv('kg.csv', low_memory=False) # KG, where we want to append the new rows
icd = pd.read_csv('Mimic IV/icd9to10_dict.csv') # Maps ICD9 / ICD10 to diseases
xrefMondoICD = pd.read_csv('xrefs.tsv', sep='\t') # Crossreferences from Mondo to various other Codes, including ICD9 and 10
diseaseFeatures = pd.read_csv('disease_features.csv', low_memory=False) #Mondocodes for PrimeKG Codes

#get only icd9 values in the xrefs
xrefMondoICD9 = xrefMondoICD[xrefMondoICD['xref'].str.startswith('ICD9')]
#Put into format that matches with ICD9 to 10 frame
xrefMondoICD9.loc[:, 'xref'] = xrefMondoICD9['xref'].str.replace('.', '', regex=False)
xrefMondoICD9.loc[:, 'xref'] = xrefMondoICD9['xref'].str.replace('ICD9:', '', regex=False)
xrefMondoICD9.columns = ['mondo', 'icd9']

#Merge ICD Table with the xref_icd9 table
xrefMondoICD9Fused = pd.merge(xrefMondoICD9, icd, on='icd9', how='inner').drop_duplicates()

#Repeat process with ICD10 xref
#get only icd10 values in the xrefs
xrefMondoICD10 = xrefMondoICD[xrefMondoICD['xref'].str.startswith('ICD10')]
#Put into format that matches with ICD9 to 10 frame
xrefMondoICD10.loc[:, 'xref'] = xrefMondoICD10['xref'].str.replace('.', '', regex=False)
xrefMondoICD10.loc[:, 'xref'] = xrefMondoICD10['xref'].str.replace('ICD10CM:', '', regex=False)
xrefMondoICD10.columns = ['mondo', 'icd10']

#Merge the ICD Table with the xref icd10 table
xrefMondoICD10Fused = pd.merge(xrefMondoICD10, icd, on='icd10', how='inner').drop_duplicates()

#Order xrefMondoICD10Fused so that the colums match the other part of the table
new_order = ['mondo', 'icd9', 'icd10', 'Description']
xrefMondoICD10Fused = xrefMondoICD10Fused[new_order]


#concat the 2 tables
xrefMondoICD9To10 = pd.concat([xrefMondoICD9Fused, xrefMondoICD10Fused], ignore_index=True).drop_duplicates()

#Remove the ICD9 Part (its useless now)
xrefMondoICD10Final = xrefMondoICD9To10[['mondo', 'icd10', 'Description']].drop_duplicates()
#Remove MONDO:0..00
xrefMondoICD10Final.loc[:, 'mondo'] = xrefMondoICD10Final['mondo'].str.replace('MONDO:', '', regex=False)
xrefMondoICD10Final.loc[:, 'mondo'] = xrefMondoICD10Final['mondo'].str.lstrip('0')
xrefMondoICD10Final.loc[:, 'mondo'] = xrefMondoICD10Final['mondo'].replace('', '0')

#Save Data
xrefMondoICD10Final.to_csv('xrefMondoICD10.csv', index=False)

#Get the diseases of PrimeKG
kgx = kg[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']]
kgxDisease = kgx[kgx['x_type'] == 'disease'].drop_duplicates()
kgxDisease.columns = ['node_index', 'y_id', 'y_type', 'y_name', 'y_source']
kgMondo = pd.merge(kgxDisease, diseaseFeatures, on='node_index', how='inner').drop_duplicates()
kgMondo = kgMondo[['node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'mondo_id']].drop_duplicates()

kgx = kg[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']]
kgxDisease = kgx[kgx['x_type'] == 'disease'].drop_duplicates()
kgxDisease.columns = ['node_index', 'y_id', 'y_type', 'y_name', 'y_source']
kgxMondo = pd.merge(kgxDisease, diseaseFeatures, on='node_index', how='inner').drop_duplicates()
kgxMondo = kgxMondo[['node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'mondo_id']].drop_duplicates()

kgy = kg[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']]
kgyDisease = kgy[kgy['y_type'] == 'disease'].drop_duplicates()
kgyDisease.columns = ['node_index', 'y_id', 'y_type', 'y_name', 'y_source']
kgyMondo = pd.merge(kgyDisease, diseaseFeatures, on='node_index', how='inner').drop_duplicates()
kgyMondo = kgyMondo[['node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'mondo_id']].drop_duplicates()

kgMondo = pd.concat([kgyMondo, kgxMondo], ignore_index=True).drop_duplicates()
kgMondo.columns = ['node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'mondo']
kgMondo['mondo'] = kgMondo['mondo'].astype(str)

#This table contains all the illnesses that PrimeHḰG and Mimic have in common
fusedDiseases = pd.merge(kgMondo, xrefMondoICD10Final, on='mondo', how='inner').drop_duplicates()
fusedDiseases = fusedDiseases[['node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'icd10']].drop_duplicates()

#Get the symptom rows
symptoms = ftd[['symp_title', 'diagnosis']].drop_duplicates()
symptoms['x_source'] = 'MIMIC-IV'
symptoms['relation'] = 'symptom_disease'
symptoms.columns = ['x_name', 'icd10', 'x_source', 'relation']

#Fuse symptoms to table
symptomDiseaseRaw = pd.merge(symptoms, fusedDiseases, on='icd10', how='inner').drop_duplicates()


#give symptoms new node_index and id
onlySymp = symptomDiseaseRaw[['x_name']].drop_duplicates()
highestIndexX = kg['x_index'].max()
highestIndexY = kg['y_index'].max()
highestIndex = max(highestIndexX, highestIndexY) #Should be 129374 in default prime kg
onlySymp['x_index'] = range(highestIndex+1, highestIndex+len(onlySymp)+1)
onlySymp['x_id'] = range(highestIndex+1, highestIndex+len(onlySymp)+1)

symptomDiseaseUnordered = pd.merge(onlySymp, symptomDiseaseRaw, on='x_name', how='inner').drop_duplicates()

#add display relation, add x_type take out the icd10 and bring in right order
symptomDiseaseUnordered['display_relation'] = 'indicative'
symptomDiseaseUnordered['x_type'] = 'symptom'
symptomDiseaseUnordered = symptomDiseaseUnordered[['x_name', 'x_index', 'x_id', 'x_source', 'relation', 'node_index', 'y_id', 'y_type', 'y_name', 'y_source', 'display_relation', 'x_type']]
symptomDiseaseUnordered.columns = [['x_name', 'x_index', 'x_id', 'x_source', 'relation', 'y_index', 'y_id', 'y_type', 'y_name', 'y_source', 'display_relation', 'x_type']]
newOrder = ['relation', 'display_relation', 'x_index', 'x_id', 'x_type', 'x_name', 'x_source', 'y_index', 'y_id', 'y_type', 'y_name', 'y_source']
symptomDiseaseFinal = symptomDiseaseUnordered[newOrder]

#concat the new edges to primekg
newKg = kg.astype(str)
symptomDiseaseFinal = symptomDiseaseFinal.astype(str)

# Flatten the MultiIndex columns into a single level
symptomDiseaseFinal.columns = [f"{level[0]}" for level in symptomDiseaseFinal.columns] # I literally have no idea why this was multiindexed previously in the first place but now it works at least :)

# Reset index to default integer index
symptomDiseaseFinal.reset_index(drop=True, inplace=True)


primeKgWithSymptoms = pd.concat([newKg, symptomDiseaseFinal], ignore_index=True).drop_duplicates()
print(primeKgWithSymptoms)

#save
#primeKgWithSymptoms.to_csv('newKg.csv', index=False)
