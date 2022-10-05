import pandas as pd
import numpy as np
from sklearn import preprocessing

#  Functions for reading, processing, and writing data from BreastCancer Wisconsin dataset.

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw data into useful files for model."""

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(drop_exact_duplicates)#
                    .pipe(print_shape, msg=' Shape after remove exact cell examples')
                    .pipe(drop_duplicates, drop_cols=['index'])
                    .pipe(print_shape  ,
                          msg=' Shape after remove index duplicates')
                    .pipe(clean_misslabeled)
                    .pipe(transform_output)
                    .pipe(drop_exact_duplicates)
                    .pipe(sort_data, col = 'diagnosis')
                    .pipe(print_shape, msg=' Shape output')
                    )

    return process_data

# function to print shape of the dataframe
def print_shape(data: pd.DataFrame, msg: str = 'Shape =') -> pd.DataFrame:
    """Print shape of dataframe."""
    print(f'{data.shape}{msg}')
    return data

def sort_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    "Sort data by and specific column"
    data = data.sort_values(by=col,ascending= False)
    return data

# remove duplicates from data based on a column
def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data=data.drop_duplicates(keep='first')
    return data

# remove duplicates from data based on a column
def drop_duplicates(data: pd.DataFrame,
                    drop_cols: list) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    data = data.drop_duplicates(subset=drop_cols, keep='first')
    return data

def transform_output(data: pd.DataFrame) -> pd.DataFrame:
    """ Replace target column to 1 and 0 values"""
    label_encoding = preprocessing.LabelEncoder()
    data['diagnosis'] = label_encoding.fit_transform(data['diagnosis'])
    return data
def clean_misslabeled(data:pd.DataFrame)->pd.DataFrame():
    """function to drop a row which target is wrongly labeled  """
    data = data[(data['diagnosis'] == 'B') | (data['diagnosis'] == 'M')]
    return data
