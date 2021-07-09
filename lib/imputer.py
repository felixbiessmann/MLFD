import pandas as pd
from pandas.util import hash_pandas_object
from lib.helpers import get_performance, df_to_ag_style
from autogluon.tabular import TabularPredictor


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

    Returns a tuple consisting of the leaderboard of the predictor object,
    Returns a tuple consisting of the runtime for training, the test dataset
    and the predictor object.
    """
    d = 'agModels'  # folder to store trained models
    hash_sum = hash_pandas_object(df_train).sum()
    checksum = hash(str(hash_sum) + str(label) + str(random_state))
    try:
        predictor = TabularPredictor.load(f'{d}/{checksum}')
    except FileNotFoundError:
        p = TabularPredictor(label=label, path=f'{d}/{checksum}')
        predictor = p.fit(train_data=df_train,
                          tuning_data=df_test,
                          verbosity=verbosity)
    return predictor


def ml_imputer(df_train,
               df_validate,
               df_test,
               label_column: str) -> pd.DataFrame:

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
