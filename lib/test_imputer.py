import pytest
import numpy as np
import pandas as pd
from lib.imputer import ml_imputer, run_ml_imputer_on_fd_set


class simpleInput:
    """A simple set of data as it is used for basic tests."""
    def __init__(self, title, continuous_cols, fd, df_train, df_validate,
                 df_test):
        self.title = title
        self.continuous_cols = continuous_cols
        self.fd = fd
        self.df_train = df_train
        self.df_validate = df_validate
        self.df_test = df_test
        self.full_df = pd.concat([self.df_train,
                                  self.df_validate,
                                  self.df_test],
                                  ignore_index=True)


@pytest.fixture
def minimal_dataset():
    title = "simple_input"
    continuous_cols = [0, 1, 2]
    fd = {2: [1, 0]}
    df_train = pd.DataFrame([[1, 2, 3], [7, 8, 15], [11, 13, 24],
                            [np.nan, np.nan, np.nan]],
                            columns=[0, 1, 2])
    df_validate = pd.DataFrame([[1, 2, 3], [4, 5, 9], [5, 7, 12]],
                               columns=[0, 1, 2])
    df_test = pd.DataFrame([[8, 9, 17], [10, 10, 20], [20, 30, 50]],
                           columns=[0, 1, 2])

    return simpleInput(title, continuous_cols, fd, df_train,
                       df_validate, df_test)


def test_ml_imputer(minimal_dataset):
    df_imputed = ml_imputer(minimal_dataset.df_train,
                            minimal_dataset.df_validate,
                            minimal_dataset.df_test,
                            str(2))

    assert df_imputed.shape[1] == 2  # consists of label and labed_imputed
    assert minimal_dataset.df_validate.shape[0] == df_imputed.shape[0]


def test_run_ml_imputer_on_fd_set(minimal_dataset):
    fds = minimal_dataset.fd
    rhs = list(fds.keys())[0]
    lhs = fds[rhs]
    fds[rhs] = [fds[rhs]]

    ml_imputer_results = run_ml_imputer_on_fd_set(
                            minimal_dataset.df_train,
                            minimal_dataset.df_validate,
                            minimal_dataset.df_test,
                            fds,
                            minimal_dataset.continuous_cols)

    assert 2 in ml_imputer_results
    assert 'lhs' in ml_imputer_results[rhs][0]
    assert ml_imputer_results[rhs][0]['lhs'] == list(map(str, lhs))
