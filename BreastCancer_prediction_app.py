from typing import Dict
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

from src.features.build_features import *

# ---------------------- Lists ---------------------------
columns_list_ready =['radius_worst', 'concavity_worst', 'fractal_dimension_worst',
       'texture_worst', 'smoothness_worst', 'symmetry_worst', 'perimeter_se',
       'smoothness_se', 'area_se', 'texture_se', 'fractal_dimension_se',
       'symmetry_se', 'symmetry_mean']

total_features = ['perimeter_se', 'radius_worst', 'concave points_mean',
            'smoothness_mean', 'area_mean', 'concavity_se', 'texture_mean',
            'concavity_worst', 'smoothness_se', 'concave points_se', 'area_worst',
            'compactness_mean', 'radius_mean', 'area_se', 'concave points_worst',
            'fractal_dimension_worst', 'perimeter_worst', 'texture_se',
            'fractal_dimension_mean', 'texture_worst', 'smoothness_worst',
            'concavity_mean', 'symmetry_mean', 'symmetry_worst',
            'fractal_dimension_se', 'perimeter_mean', 'compactness_worst',
            'symmetry_se', 'compactness_se', 'radius_se']


# ---------------------- Prediction Model ------------------------------

model = load("models/NB_final_model.joblib")

#Caching the model for faster loading
@st.cache
# list of inputs are too long so i use *args instead
def predict(*args):
    list_variables = list(args)
    data_input = pd.DataFrame([list_variables],columns= columns_list_ready)
    data_predict = feature_process(data_input)

    prediction = model.predict(data_predict)
    return prediction

def feature_process(data: pd.DataFrame) -> pd.DataFrame:

    """Process raw data into useful files for model."""
    data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(data_tostring)
                    .pipe(clean_blankspaces, cols_to_clean=data.columns)
                    .pipe(na_count)
                    .pipe(replace_missing_values, replace_values=['rxctf378968 7656463sdfg', '-88888765432345.0', '999765432456788.0', '?'])
                    .pipe(na_count)
                    .pipe(data_tofloat)
                    .pipe(outlier_tona)
                    .pipe(na_count)
                    .pipe(to_category)
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after droping duplicates')
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after dropping unnecesary cols')
                    .pipe(drop_hc_cols,msg='Dropping highly correlated features')
                    )
    return data


    # Ensure the order of column in the test set is in the same order than in train set
    data = data[columns_list_ready]
    return data

# --------------- streamlit app ------------------------------------
st.title('Breast  mass malignant cells clasiffication model ')
st.image("""https://storage.googleapis.com/kaggle-datasets-images/3724/5903/a8a637953c923bf989852df53b54d769/dataset-card.jpg""")
st.header('Insert the patient information')
"""race = st.radio('Race:', ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'], horizontal = True)
age = st.radio('Age:', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
       '[60-70)', '[70-80)', '[80-90)', '[90-100)'], horizontal = True)
admission_type_id = st.number_input('Admission Type:',min_value=1, max_value=8, value=1)
discharge_disposition_id = st.number_input('Discharge disposition:',min_value=1, max_value=28, value=1)
admission_source_id = st.number_input('Admission Source:',min_value=1, max_value=25, value=1)
time_in_hospital = st.number_input('time in hospital:',min_value=1, max_value=100, value=1)
medical_specialty = st.selectbox('Medical Specialty:', med_specialty)
num_lab_procedures = st.number_input('# lab Procedures:',min_value=0, max_value=100, value=0)
num_procedures = st.number_input('# Procedures:',min_value=0, max_value=100, value=0)
num_medications = st.number_input('# Medications:',min_value=0, max_value=100, value=0)
number_outpatient = st.number_input('# outpatient visits:',min_value=0, max_value=100, value=0)
number_emergency = st.number_input('# of emergency visits:',min_value=0, max_value=100, value=0)
number_inpatient = st.number_input('# inpatient visits:',min_value=0, max_value=100, value=0)
diag_1 = st.selectbox('Diagnosis 1', diag_1_list)
diag_2 = st.selectbox('Diagnosis 2', diag_2_list)
diag_3 = '250' #random
number_diagnoses = st.number_input('# of diagnosis:',min_value=1, max_value=100, value=1)
max_glu_serum = st.radio('Glucose serum test result:', ['None', '>300', 'Norm', '>200'], horizontal = True)
A1Cresult = st.radio('A1c test result:', ['None', '>8', '>7', 'Norm' ], horizontal = True)
metformin = st.radio('metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
repaglinide = st.radio('repaglinide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
nateglinide = st.radio('nateglinide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
chlorpropamide = st.radio('chlorpropamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glimepiride = st.radio('glimepiride', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
acetohexamide = st.radio('acetohexamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glipizide = st.radio('glipizide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glyburide = st.radio('glyburide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
tolbutamide = st.radio('tolbutamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
pioglitazone = st.radio('pioglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
rosiglitazone = st.radio('rosiglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
acarbose = st.radio('acarbose', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
miglitol = st.radio('miglitol', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
troglitazone = st.radio('troglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
tolazamide = st.radio('tolazamide', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
insulin = st.radio('insulin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glyburide_metformin = st.radio('glyburide-metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
glipizide_metformin = st.radio('glipizide-metformin', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
metformin_rosiglitazone = st.radio('metformin-rosiglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)
metformin_pioglitazone = st.radio('metformin-pioglitazone', ['No', 'Up', 'Steady', 'Down'], horizontal = True)

change = st.radio('Chagne of medications', ['No', 'Ch'] , horizontal = True)
diabetesMed = st.radio('Diabetes medications', ['no', 'yes'], horizontal = True)"""
""""#columns to drop
encounter_id = 54500028 #random number
patient_nbr = 3851154 #random number
gender = 'Female' # random
payer_code = 'MC' # random
examide = 'No' # random
citoglipton = 'No' # random
glimepiride_pioglitazone = 'No' # random
weight = 'No' # random


#data_input = dict(zip(columns_list,columns_variables))

if st.button('Predict Readmission'):
    readmission = predict(
            encounter_id,
            patient_nbr,
            race,
            gender,
            age,
            weight,
            admission_type_id,
            discharge_disposition_id,
            admission_source_id,
            time_in_hospital,
            payer_code,
            medical_specialty,
            num_lab_procedures,
            num_procedures,
            num_medications,
            number_outpatient,
            number_emergency,
            number_inpatient,
            diag_1,
            diag_2,
            diag_3,
            number_diagnoses,
            max_glu_serum,
            A1Cresult,
            metformin,
            repaglinide,
            nateglinide,
            chlorpropamide,
            glimepiride,
            acetohexamide,
            glipizide,
            glyburide,
            tolbutamide,
            pioglitazone,
            rosiglitazone,
            acarbose,
            miglitol,
            troglitazone,
            tolazamide,
            examide,
            citoglipton,
            insulin,
            glyburide_metformin,
            glipizide_metformin,
            glimepiride_pioglitazone,
            metformin_rosiglitazone,
            metformin_pioglitazone,
            change,
            diabetesMed
    )
    st.success(f'The prediction tells that the patient will have a readmission in less than 30 days: {bool(readmission)}')"""