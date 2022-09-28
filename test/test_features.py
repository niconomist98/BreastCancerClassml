import pytest
import pandas as pd
import numpy as np

#import created functions
from src.features.build_features import *


#load data
@pytest.fixture
def leer_datos():
    data_test = pd.read_csv('../data/interim/x_train.csv')
    return data_test

@pytest.fixture
def leer_target():
    target_test= pd.read_csv('../data/interim/y_train.csv')
    return target_test

def test_data(leer_datos):
    columnas_name =['index', 'perimeter_se', 'radius_worst', 'concave points_mean',
       'smoothness_mean', 'area_mean', 'concavity_se', 'texture_mean',
       'concavity_worst', 'smoothness_se', 'concave points_se', 'area_worst',
       'compactness_mean', 'radius_mean', 'area_se', 'concave points_worst',
       'iuytr', 'fractal_dimension_worst', 'perimeter_worst', 'texture_se',
       'fractal_dimension_mean', 'texture_worst', 'smoothness_worst',
       'concavity_mean', 'id', 'symmetry_mean', 'symmetry_worst', 'erty',
       'fractal_dimension_se', 'perimeter_mean', 'compactness_worst',
       'symmetry_se', 'compactness_se', 'radius_se']
    columnas = leer_datos.columns
    assert len(columnas) == 34
    assert set(columnas_name) == set(columnas)

def test_encoding_target(leer_target):
    assert list(leer_target['diagnosis'].unique()) == [0,1]


def test_data_tostring(leer_datos):
    datastring=data_tostring(leer_datos)
    a=pd.Series(datastring.dtypes=='string').sum() ##ammount of object type columns
    b=pd.Series(datastring.columns).value_counts().sum() ##total ammount of columns
    assert a==b


def test_outlier_tona(leer_datos):
    data = data_tostring(leer_datos)
    data = clean_blankspaces(data, data.columns)
    data = replace_missing_values(data, ['rxctf378968 7656463sdfg', '-88888765432345.0', '999765432456788.0', '?'])
    data=data_tofloat(data)
    nasn = data.isna().sum().sum()
    data=outlier_tona(data)
    nasn2 = data.isna().sum().sum()
    assert nasn2>nasn

def test_data_tofloat(leer_datos):
    data = data_tostring(leer_datos)
    data = clean_blankspaces(data, data.columns)
    data = replace_missing_values(data, ['rxctf378968 7656463sdfg', '-88888765432345.0', '999765432456788.0', '?'])
    data = data_tofloat(data)
    a=pd.Series(data.dtypes=='float64').sum() ##ammount of object type columns
    b=pd.Series(data.columns).value_counts().sum() ##total ammount of columns
    assert a==b
def test_imputer_KNN(leer_datos):
    data=data_tostring(leer_datos)
    data=clean_blankspaces(data,data.columns)
    data=replace_missing_values(data,['rxctf378968 7656463sdfg', '-88888765432345.0', '999765432456788.0', '?'])
    data=data_tofloat(data)
    data=outlier_tona(data)
    data=imputer_KNN(data)

    assert data.isna().sum().sum()==0








