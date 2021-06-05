import random
import pytest
import tempfile
import itertools
import numpy as np
import pandas as pd
import lib.library as library


class simpleInput:
    """A simple set of data as it is used for basic tests."""
    def __init__(self, title, continuous_cols, fd, df_train, df_validate, df_test):
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


def test_get_performance():
    source = {'x': [1, 2, 3], 'x_imputed': [1, 2, 0]}
    df_source = pd.DataFrame(source)
    p = library.get_performance(df_source, 'x', [], [])
    assert p['lhs'] == []
    assert round(p['f1'], 3) == 0.667
    assert round(p['recall'], 3) == 0.667
    assert round(p['precision'], 3) == 0.667


def test_random_dependency_generator():
    n = random.randint(0, 800)
    cols = list(range(0, 30))

    dependencies = library.random_dependency_generator(cols, n)
    for rhs in dependencies:
        lhs = dependencies[rhs]
        lhs.sort()
        no_lhs = len(lhs)
        no_unique_lhs = len(list(lhs for lhs, _ in itertools.groupby(lhs)))
        assert no_lhs == no_unique_lhs
    no_dependencies = sum([len(dependencies[rhs]) for rhs in dependencies])
    assert no_dependencies <= n


def test_split_df():
    pass


def test_read_fds():
    fds = library.read_fds('lib/test_data/test_fd.txt')
    rhs = [x for x in fds]
    assert min(rhs) == 1  # index is not a rhs
    assert max(rhs) == (len(rhs))  # same reason as above

    lhs_flat = [z for x in fds for y in fds[x] for z in y]
    assert min(lhs_flat) == 0
    assert max(lhs_flat) == (len(rhs))


def test_split_df_and_load_dataframes(minimal_dataset, tmp_path):
    tmp_path_str = str(tmp_path)
    library.split_df(
        minimal_dataset.title,
        minimal_dataset.full_df,
        [0.33, 0.33, 0.33],
        tmp_path_str+'/')

    df_train, df_validate, df_test = library.load_dataframes(
        tmp_path_str+'/', minimal_dataset.title, 'noData')

    df_glued = pd.concat([df_train, df_validate, df_test])

    assert sum(df_glued.duplicated()) == 0

    # check if missing value deletion from load_dataframes works
    assert df_glued.isna().values.sum() == 0
    assert df_glued.shape[0] <= minimal_dataset.full_df.shape[0]


def test_ml_imputer(minimal_dataset):
    df_imputed = library.ml_imputer(minimal_dataset.df_train,
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

    ml_imputer_results = library.run_ml_imputer_on_fd_set(
                            minimal_dataset.df_train,
                            minimal_dataset.df_validate,
                            minimal_dataset.df_test,
                            fds,
                            minimal_dataset.continuous_cols)

    assert 2 in ml_imputer_results
    assert 'lhs' in ml_imputer_results[rhs][0]
    assert ml_imputer_results[rhs][0]['lhs'] == list(map(str, lhs))
