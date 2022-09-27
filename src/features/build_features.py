# feature encodign para antes de ML
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.impute import KNNImputer

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

    tipificado = StandardScaler().fit(x_train)##Creating a scaler
    x_train = pd.DataFrame(tipificado.transform(x_train),columns=x_train.columns)##scaling x_train


    x_train=imputer_KNN(x_train)## imputing using KNN method

    y_train = process_data["diagnosis"]

    x_train.to_csv(f'{output_filepath}/x_train_model_input.csv', index=False)
    y_train.to_csv(f'{output_filepath}/y_train_model_input.csv', index=False)
    # End
    print(f' number of nas {x_train.isna().sum().sum()}')





def imputer_KNN (X_train:pd.DataFrame)->pd.DataFrame:
    """Use knn to impute na values """
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    return X_train


##function to count nas
def na_count(data:pd.DataFrame)->pd.DataFrame:
    """count  total nas in a dataframe"""
    nulls=data.isnull().sum().sum()
    print(f' la cantidad de nulos es : {nulls}')
    return data


##function to transform outliers in nas
def outlier_tona(data:pd.DataFrame)->pd.DataFrame:
    """convert outliers to na"""
    for i in data.columns:
        Q1 = data[i].quantile(0.25)
        Q3 = data[i].quantile(0.75)
        IQR = Q3 - Q1
        data[i] = np.where((data[i] < (Q1 - 1.5 * IQR)) | (data[i] > (Q3 + 1.5 * IQR)), np.nan, data[i])

    return data
##function to transform target to category
def to_category(data:pd.DataFrame)->pd.DataFrame:
    """Convert a colum to category"""
    data['diagnosis'].astype('category')
    return data


# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

#Function to convert columns of a df to float type
def data_tofloat(data:pd.DataFrame)->pd.DataFrame:
    """Change dataset columns dtype"""
    data=data.astype('float')
    return data
#function to convert columns of a df to stirng type

def data_tostring(data:pd.DataFrame)->pd.DataFrame:
    """Convert dataset columns to type string """
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

# function to replace values with np.nan
def replace_missing_values(data: pd.DataFrame,
                           replace_values: list) -> pd.DataFrame:
    """Replace missing values in data with np.nan"""
    data=data.replace(replace_values, np.nan)
    return data

#Function to clean blankspaces after each string of the dataset
def clean_blankspaces(data:pd.DataFrame,cols_to_clean:list)-> pd.DataFrame:
    """Function to delete blankspaces after and before each string of the dataset"""
    for i in cols_to_clean:
        data[i] = data[i].str.strip()
    return data


def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)







if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/interim', f'{project_dir}/data/processed')