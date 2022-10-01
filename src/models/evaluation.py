import pandas as pd
import logging
from joblib import load
from sklearn.metrics import recall_score
from pathlib import Path
import numpy as np

# libraries to import function from other folder
import sys
import os


sys.path.append(os.path.abspath('src/'))

from src.features.build_features import *


def main(input_filepath, output_filepath, input_test_filepath, report_filepath):
    """ Runs model training scripts to turn processed data from (../processed) into
        a machine learning model (saved in ../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluating NB model')

    model = load(f'{output_filepath}/NB_final_model.joblib')

    x_train = pd.read_csv(f"{input_filepath}/x_train_model_input.csv")
    y_train = pd.read_csv(f"{input_filepath}/y_train_model_input.csv")

    y_pred = model.predict(x_train)

    train_score = recall_score(y_train, y_pred)
    print(f"Train Score: {train_score}")

    with open(f'{report_filepath}/train_score.txt', 'w') as f:
        f.write(f"Train reacall Score: {train_score}")

    # test predictions

    x_test = pd.read_csv(f"{input_test_filepath}/x_test.csv")
    y_test = pd.read_csv(f"{input_test_filepath}/y_test.csv")

    test = pd.concat([x_test, y_test], axis=1)

    test_eval = feature_process(test)

    x_test_model = test_eval.drop('diagnosis', axis=1)
    y_test_model = test_eval['diagnosis']

    y_test_pred = model.predict(x_test_model)

    test_eval.to_csv(f"{input_filepath}/x_test_model_input.csv")
    np.savetxt(f"{input_filepath}/y_test_pred.csv", y_test_pred, delimiter=",")

    test_score = recall_score(y_test_model, y_test_pred)
    print(f"Test Score: {test_score}")

    with open(f'{report_filepath}/test_score.txt', 'w') as f:
        f.write(f"Test recall Score: {test_score}")

cols_to_drop = ['erty','iuytr','id','index']

def feature_process(data: pd.DataFrame) -> pd.DataFrame:

    process_data = (data
                    .pipe(print_shape, msg=' Shape original')
                    .pipe(data_tostring)
                    .pipe(clean_blankspaces, cols_to_clean=data.columns)
                    .pipe(na_count)
                    .pipe(replace_missing_values,
                          replace_values=['rxctf378968 7656463sdfg', '-88888765432345.0', '999765432456788.0', '?'])
                    .pipe(na_count)
                    .pipe(data_tofloat)
                    .pipe(outlier_tona)
                    .pipe(na_count)
                    .pipe(to_category)
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after droping duplicates')
                    .pipe(drop_cols, drop_cols=cols_to_drop)
                    .pipe(drop_exact_duplicates)
                    .pipe(print_shape, msg=' Shape after dropping unnecesary cols')
                    .pipe(drop_hc_cols, msg='Dropping highly correlated features')
                    )
    return process_data


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(f'{project_dir}/data/processed',
         f'{project_dir}/models',
         f'{project_dir}/data/interim',
         f'{project_dir}/reports')

