import datawig
import pandas as pd
from hashlib import sha256
from pandas.util import hash_pandas_object
from lib.helpers import get_performance, df_to_ag_style
from autogluon.tabular import TabularPredictor
import logging
from typing import Tuple
import numpy as np


def calculate_model_hash(df, label, random_state) -> str:
    """ Calculate the hash of a dataframe and label."""
    logger = logging.getLogger('pfd')
    hash_sum = hash_pandas_object(df).sum()
    m = sha256()
    m.update(bytes(str(hash_sum) + str(label) + str(random_state), 'utf-8'))
    checksum = m.hexdigest()
    logger.info(f'Calculated a data-checksum of {checksum}.')
    return checksum


def train_cleaning_model(df_dirty: pd.DataFrame,
                         label: int,
                         random_state: int = 0,
                         **kwargs) -> Tuple[datawig.AutoGluonImputer, str]:
    """
    Train an autogluon model for the purpose of cleaning data.
    Optionally, you can set verbosity to control how much output AutoGluon
    produces during training.

    The function caches models that have been trained on the same data. If
    you wish to prevent this, set random_state at random.

    Returns a tuple (predictor object, model_checksum).
    """
    lhs = list(df_dirty.columns)
    del lhs[label - 1]
    logger = logging.getLogger('pfd')
    checksum = calculate_model_hash(df_dirty, label, random_state)

    imputer = datawig.AutoGluonImputer(
        model_name=checksum,
        input_columns=lhs,
        output_column=label,
        force_multiclass=kwargs['force_multiclass'],
        verbosity=kwargs['verbosity'],
        label_count_threshold=kwargs['label_count_threshold']
    )

    try:
        if kwargs.get('force_retrain'):
            logger.info("Retraining a model as force_retrain is set.")
            raise FileNotFoundError
        imputer = datawig.AutoGluonImputer.load(output_path='./',
                                                model_name=checksum)
    except FileNotFoundError:
        logger.info("Didn't find a model to load from cache.")
        imputer.fit(train_df=df_dirty,
                    holdout_frac=kwargs.get('holdout_frac', None),
                    time_limit=kwargs['time_limit'],
                    hyperparameters=kwargs.get('hyperparameters', None),
                    hyperparameter_tune_kwargs=kwargs.get('hyperparameter_tune_kwargs', None),
                    # preset='best_quality'
                    )
        imputer.save()
    return imputer, checksum


def make_cleaning_prediction(df_dirty,
                             probas,
                             target_col) -> pd.DataFrame:
    """
    When cleaning, the task is defined such that the ids of the dirty rows
    are known. When cleaning, it is thus always incorrect to return the class
    that is known to be dirty.

    The strategy I use here is:
    1) Check the most probable class. If the class is not the dirty class,
    return it.
    2) If the returned class is the dirty class, return the second most likely
    class.
    """
    dirty_label_mask = pd.DataFrame()
    for c in probas.columns:
        dirty_label_mask[c] = df_dirty[target_col] == c
    probas[dirty_label_mask] = -1  # dirty labels are never chosen

    i_classes = np.argmax(probas.values, axis=1)
    cleaning_predictions = [probas.columns[i] for i in i_classes]
    return pd.Series(cleaning_predictions)


def train_model(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                label: str,
                verbosity: int = 0,
                random_state: int = 0) -> TabularPredictor:
    """
    Train an autogluon model for df_train, df_test. Specify the label column.
    Optionally, you can set verbosity to control how much output AutoGluon
    produces during training.

    The function caches models that have been trained on the same data by
    computing the hash of df_train and comparing that to existing models.

    Returns the predictor object.

    TODO: Optimize this bad boy for experiments. Would be k-fold
    cross-validation instead of train-test split and a AG-preset that opts
    for highest quality model. Also no or very high time_limit.
    """
    logger = logging.getLogger('pfd')
    d = 'agModels'  # folder to store trained models
    checksum = calculate_model_hash(df_train, label, random_state)
    model_path = f'{d}/{checksum}'
    logger.info(f'Calculated a checksum of {checksum}.')
    try:
        predictor = TabularPredictor.load(model_path)
    except FileNotFoundError:
        logger.info("Didn't find a model to load from the cache.")
        p = TabularPredictor(label=label, path=model_path)
        predictor = p.fit(train_data=df_train,
                          tuning_data=df_test,
                          time_limit=20,
                          verbosity=verbosity,
                          presets='medium_quality_faster_train')
    return predictor


def ml_imputer(df_train,
               df_validate,
               df_test,
               label_column: str) -> pd.DataFrame:
    """
    Trains a model and imputes values in df_validate.
    Returns a DataFrame with two columns, df_validate and df_imputed.
    """

    train_data = df_to_ag_style(df_train)
    test_data = df_to_ag_style(df_test)
    validate_data = df_to_ag_style(df_validate)

    predictor = train_model(train_data, test_data, str(label_column))

    validate_data_no_y = validate_data.drop(labels=[label_column], axis=1)
    y_pred = predictor.predict(validate_data_no_y)
    print("These are the predicted labels: ")
    print(y_pred)
    print("And this is the validation-dataset: ")
    print(validate_data)

    df_imputed = y_pred.to_frame(name=f'{label_column}_imputed')
    print('df_imputed: ')
    print(df_imputed)

    df_validate_imputed = pd.merge(df_validate.loc[:, label_column],
                                   df_imputed,
                                   left_index=True,
                                   right_index=True,
                                   how='left')
    return df_validate_imputed


def run_ml_imputer_on_fd_set(df_train, df_validate, df_test, fds,
                             continuous_cols=[]):
    """ Runs ml_imputer for every fd contained in a dictionary on a split df.

    Executes ml_imputer() for every fd in fds. df_train and df_split should be
    created using split_df(). continuous_cols is used to distinguish between
    columns containing continuous_cols numbers and classifiable objects.

    Keyword arguments:
        df_train -- a df on which fds have been detected.
        df_validate -- a df on which no fds have been detected.
        fds -- a dictionary with an fd's possible rhs as values and a list of
        lhs-combinations as value of fds[rhs].

    Returns:
    A dictionary consisting of rhs's as keys with associated performance
    metrics for each lhs. Performance metrics are precision, recall and
    f1-measure for classifiable data and the mean squared error as well as
    the number of unsucessful imputations for continuous data.
    """
    ml_imputer_results = {}
    for rhs in fds:
        rhs_results = []
        for lhs in fds[rhs]:
            relevant_cols = lhs + [rhs]
            df_subset_train = df_train.iloc[:, relevant_cols]
            df_subset_validate = df_validate.iloc[:, relevant_cols]
            df_subset_test = df_test.iloc[:, relevant_cols]

            ml_result = ml_imputer(df_subset_train,
                                   df_subset_validate,
                                   df_subset_test,
                                   str(rhs))

            result = get_performance(ml_result,
                                     rhs, lhs, continuous_cols)
            rhs_results.append(result)
        ml_imputer_results[rhs] = rhs_results
    return ml_imputer_results
