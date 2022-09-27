# feature encodign para antes de ML
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def main(input_filepath, output_filepath):
    """ Runs data feature engineering scripts to turn interim data from (../interim) into
        cleaned data ready for machine learning (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data')

    x = pd.read_csv(f"{input_filepath}/x_train.csv")
    y = pd.read_csv(f"{input_filepath}/y_train.csv")

    data = pd.concat([x, y], axis=1)

    """Process raw data into useful files for model."""
    cols_to_drop = ['erty','iuytr','id','index']

    medication = ['metformin', 'repaglinide', 'nateglinide',
                  'chlorpropamide', 'glimepiride', 'glipizide',
                  'glyburide', 'pioglitazone', 'rosiglitazone',
                  'acarbose', 'miglitol', 'insulin',
                  'glyburide-metformin', 'tolazamide',
                  'metformin-pioglitazone', 'metformin-rosiglitazone',
                  'glipizide-metformin', 'troglitazone', 'tolbutamide',
                  'acetohexamide']

    med_specialty = ['Unknown', 'InternalMedicine', 'Family/GeneralPractice',
                     'Cardiology', 'Surgery-General', 'Orthopedics', 'Gastroenterology',
                     'Nephrology', 'Orthopedics-Reconstructive',
                     'Surgery-Cardiovascular/Thoracic', 'Pulmonology', 'Psychiatry',
                     'Emergency/Trauma', 'Surgery-Neuro', 'ObstetricsandGynecology',
                     'Urology', 'Surgery-Vascular', 'Radiologist']


    features= ['perimeter_se', 'radius_worst', 'concave points_mean',
            'smoothness_mean', 'area_mean', 'concavity_se', 'texture_mean',
            'concavity_worst', 'smoothness_se', 'concave points_se', 'area_worst',
            'compactness_mean', 'radius_mean', 'area_se', 'concave points_worst',
            'fractal_dimension_worst', 'perimeter_worst', 'texture_se',
            'fractal_dimension_mean', 'texture_worst', 'smoothness_worst',
            'concavity_mean', 'symmetry_mean', 'symmetry_worst',
            'fractal_dimension_se', 'perimeter_mean', 'compactness_worst',
            'symmetry_se', 'compactness_se', 'radius_se']

    process_data = (data
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
                    .pipe(drop_cols,drop_cols=cols_to_drop)
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after dropping unnecesary cols')

                    )

    x_train = process_data.drop("diagnosis", axis=1)
    y_train = process_data["diagnosis"]

    x_train.to_csv(f'{output_filepath}/x_train_model_input.csv', index=False)
    y_train.to_csv(f'{output_filepath}/y_train_model_input.csv', index=False)
    # End




from sklearn.model_selection import train_test_split # Import train_test_split function
def imputer_na (data:pd.DataFrame)->pd.DataFrame:
    tipificado = StandardScaler().fit(data[~data['diagnosis']])
    data = tipificado.transform(data[~data['diagnosis']])
    return data


def imputer_na (data:pd.DataFrame)->pd.DataFrame:

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_train_imputed

    return data


##function to count nas
def na_count(data:pd.DataFrame)->pd.DataFrame:
    nulls=data.isnull().sum().sum()
    print(f' la cantidad de nulos es : {nulls}')
    return data


##function to transform outliers in nas
def outlier_tona(data:pd.DataFrame)->pd.DataFrame:

    for i in data.columns:
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3 - Q1
        data[i] = np.where((data[i] < (Q1 - 1.5 * IQR)) | (data[i] > (Q3 + 1.5 * IQR)), np.nan, data[i])

    return data
##function to transform target to category
def to_category(data:pd.DataFrame)->pd.DataFrame:
    data['diagnosis'].astype('category')
    return data


# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data
def data_tofloat(data:pd.DataFrame)->pd.DataFrame:
    """Change dataset columns dtype"""
    data=data.astype('float')
    return data
def data_tostring(data:pd.DataFrame)->pd.DataFrame:
    data = data.astype('string')
    return data

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(subset=drop_cols, keep=False)


# funtion to remove columns from data
def drop_cols(data: pd.DataFrame,
              drop_cols: list = None) -> pd.DataFrame:
    """Drop columns from data."""
    return data.drop(drop_cols, axis=1)


def to_categorical(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    for x in categorical_cols:
        data[x] = data[x].astype('category')
    return data

##function to drop rows with na
def dropna(data:pd.DataFrame)-> pd.DataFrame:
    data.dropna(inplace=True)
    return data

# function to filter data from a column if is in a list of values
def filter_cols_values(data: pd.DataFrame,
                       filter_col: str,
                       filter_values: list) -> pd.DataFrame:
    """Filter columns from data."""
    data = data[~data[filter_col].isin(filter_values)]
    return data


# function to replace ? values with np.nan
def replace_missing_values(data: pd.DataFrame,
                           replace_values: list) -> pd.DataFrame:
    """Replace missing values in data with np.nan"""
    data=data.replace(replace_values, np.nan)
    return data

# function to fillna with a string in a column
def fill_na_with_string(data: pd.DataFrame,
                        fill_col: str,
                        fill_string: str) -> pd.DataFrame:
    """Fill missing values in data with a string."""
    data[fill_col] = data[fill_col].fillna(fill_string, axis=0)
    return data


# function to fill na values in one column using the values from another column
def fill_na_with_col(data: pd.DataFrame,
                     fill_col: str,
                     fill_col_from: str) -> pd.DataFrame:
    """Fill missing values in data with values from another column."""
    data[fill_col] = data[fill_col].fillna(data[fill_col_from], axis=0)
    return data


def encoding_columns(data: pd.DataFrame) -> pd.DataFrame:
    data["max_glu_serum"].replace({'>200': 1,
                                   '>300': 1,
                                   'Norm': 0,
                                   'None': -99},
                                  inplace=True)

    data["A1Cresult"].replace({'>7': 1,
                               '>8': 1,
                               'Norm': 0,
                               'None': -99},
                              inplace=True)
    data['change'].replace('Ch', 1, inplace=True)
    data['change'].replace('No', 0, inplace=True)
    data['gender'].replace('Male', 1, inplace=True)
    data['gender'].replace('Female', 0, inplace=True)
    data['diabetesMed'].replace('Yes', 1, inplace=True)
    data['diabetesMed'].replace('No', 0, inplace=True)
    return data

#Function to clean blankspaces before and after each datacell -nico
def clean_blankspaces(data:pd.DataFrame,cols_to_clean:list)-> pd.DataFrame:
    for i in cols_to_clean:
        data[i] = data[i].str.strip()
    return data


def medication_changes(data: pd.DataFrame, keys: list) -> pd.DataFrame:
    """
    Medication change for diabetics upon admission has been shown in this research:
    [What are Predictors of Medication Change and Hospital Readmission in Diabetic Patients?](https://www.ischool.berkeley.edu/projects/2017/what-are-predictors-medication-change-and-hospital-readmission-diabetic-patients)
    to be associated with lower readmission rates.
    New variable is created  to count how many changes were made in total for each patient.
    Args:
        data (pd.DataFrame):
    Returns:
        pd.DataFrame: dataframe with new column
    """

    for col in keys:
        colname = str(col) + 'temp'
        data[colname] = data[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)
        data['numchange'] = 0
    for col in keys:
        colname = str(col) + 'temp'
        data['numchange'] = data['numchange'] + data[colname]
        del data[colname]
    return data


def medication_encoding(data: pd.DataFrame, keys: list) -> pd.DataFrame:
    for col in keys:
        data[col].replace({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1},
                          inplace=True)
    return data


def diagnose_encoding(data: pd.DataFrame) -> pd.DataFrame:
    diag_cols = ['diag_1', 'diag_2']
    df_copy = data[diag_cols].copy()
    for col in diag_cols:
        df_copy[col] = df_copy[col].str.replace('E', '-')
        df_copy[col] = df_copy[col].str.replace('V', '-')
        condition = df_copy[col].str.contains('250')
        df_copy.loc[condition, col] = '250'

    df_copy[diag_cols] = df_copy[diag_cols].astype(float)
    for col in diag_cols:
        df_copy['temp'] = np.nan

        condition = (df_copy[col] >= 390) & (df_copy[col] <= 459) | (df_copy[col] == 785)
        df_copy.loc[condition, 'temp'] = 'Circulatory'

        condition = (df_copy[col] >= 460) & (df_copy[col] <= 519) | (df_copy[col] == 786)
        df_copy.loc[condition, 'temp'] = 'Respiratory'

        condition = (df_copy[col] >= 520) & (df_copy[col] <= 579) | (df_copy[col] == 787)
        df_copy.loc[condition, 'temp'] = 'Digestive'

        condition = (df_copy[col] >= 800) & (df_copy[col] <= 999)
        df_copy.loc[condition, 'temp'] = 'Injury'

        condition = (df_copy[col] >= 710) & (df_copy[col] <= 739)
        df_copy.loc[condition, 'temp'] = 'Muscoloskeletal'

        condition = (df_copy[col] >= 580) & (df_copy[col] <= 629) | (df_copy[col] == 788)
        df_copy.loc[condition, 'temp'] = 'Genitourinary'

        condition = (df_copy[col] >= 140) & (df_copy[col] <= 239) | (df_copy[col] == 780)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] >= 240) & (df_copy[col] <= 279) | (df_copy[col] == 781)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] >= 680) & (df_copy[col] <= 709) | (df_copy[col] == 782)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] >= 790) & (df_copy[col] <= 799) | (df_copy[col] == 784)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] >= 1) & (df_copy[col] <= 139)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] >= 290) & (df_copy[col] <= 319)
        df_copy.loc[condition, 'temp'] = 'Neoplasms'

        condition = (df_copy[col] == 250)
        df_copy.loc[condition, 'temp'] = 'Diabetes'

        df_copy['temp'] = df_copy['temp'].fillna('Others')
        condition = df_copy['temp'] == '0'
        df_copy.loc[condition, 'temp'] = np.nan
        df_copy[col] = df_copy['temp']
        df_copy.drop('temp', axis=1, inplace=True)
    data[diag_cols] = df_copy.copy()
    return data


def process_medical_specialty(data: pd.DataFrame, keys: list) -> pd.DataFrame:
    """
    specialties with few values will be converted to value = `other`
    """

    data.loc[~data['medical_specialty'].isin(keys), 'medical_specialty'] = 'Other'

    return data


def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)


def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    cat_cols = list(data.select_dtypes('object').columns)
    for col in cat_cols:
        data = pd.concat([data.drop(col, axis=1),
                          pd.get_dummies(data[col], prefix=col, drop_first=True)],
                         axis=1)
    return data




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/interim', f'{project_dir}/data/processed')