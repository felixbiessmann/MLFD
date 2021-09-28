import os
import pickle
from typing import List, Any, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import random


def cleaning_performance(y_clean: pd.Series, y_predicted: pd.Series):
    """
    Calculate the f1-score between the clean labels and the predicted
    labels.
    """
    y_delta = y_clean == y_predicted
    y_true = pd.Series([True for _ in y_delta])
    return f1_score(y_true, y_delta)


def error_detection_performance(y_clean: pd.Series,
                                y_predicted: pd.Series,
                                y_dirty: pd.Series):
    """
    Calculate the f1-score for finding the correct position of errors in
    y_dirty.
    """
    y_error_position_true = y_clean != y_dirty
    y_error_position_pred = y_clean != y_predicted
    return f1_score(y_error_position_true, y_error_position_pred)


def subset_df(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    exclude_cols = [int(x) for x in exclude_cols]
    new_cols = [x for x in list(df.columns) if x not in exclude_cols]
    return df.loc[:, new_cols]


def save_pickle(obj, path):
    """ Pickles object obj and saves it to path. If path doesn't exist,
    creates path. """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    pickle.dump(obj, open(path, "wb"))
    message = '{0} successfully saved to {1}!'.format(
        os.path.basename(path).split('.')[0],
        path)
    print(message)


def split_df(df, split_ratio, random_state):
    """
    Splits a dataframe into train-, validate - and test-subsets.
    If a data_path is provided, splits will be saved to the harddrive.

    Returns a tuple(df_train, df_validate, df_test) if the data_path
    is an empty string.

    Keyword arguments:
    df -- a pandas data frame
    split_ratio -- train/test/validate set ratio, e.g. [0.8, 0.1, 0.1]
    random_state -- an integer that ensures that splitting is reproducible
    across clean and dirty data
    """
    train_ratio, validate_ratio, test_ratio = split_ratio

    rest_df, test_df = train_test_split(df,
                                        test_size=test_ratio,
                                        random_state=random_state)
    train_df, validate_df = train_test_split(rest_df,
                                             test_size=validate_ratio,
                                             random_state=random_state)
    return(train_df, validate_df, test_df)


def save_dfs(save_dict: Dict, data):
    """
    Saves dataframes to the disc.

    Keyword arguments:
    save_dict -- dict of data_name: DataFrame entries
    data -- the dataset whose data is saved to the disc
    """
    import os
    import logging
    logger = logging.getLogger('pfd')

    for name, df in save_dict.items():
        path = f'{data.splits_path}{name}/'
        if not os.path.exists(path):
            os.mkdir(path)
        try:
            save_path = f'{path}{data.title}_{name}.csv'
            df.to_csv(save_path, sep=',',
                      index=False, header=None)
            logger.info(f'{name} set successfully written to {save_path}.')
        except TypeError:
            logger.error("Something went wrong saving the splits.")
    pass


def check_split_for_duplicates(list_of_dfs):
    """ Prints and returns number of duplicate entries in a list of dfs """
    no_duplicates = sum(pd.concat(list_of_dfs).duplicated())
    print(str(no_duplicates) + ' duplicates detected in the splittings.')
    return no_duplicates


def load_original_data(data, load_dirty=False):
    """
    Loads the original dataframe. Missing values are replaced with
    np.nan. If load_dirty is set to True, the dirty dataset of a
    cleaning experiment is loaded. Otherwise, the default clean dataset
    is loaded.
    """
    if load_dirty:
        df = pd.read_csv(data.dirty_data_path,
                         sep=data.original_separator,
                         header=None)
    else:
        df = pd.read_csv(data.data_path,
                         sep=data.original_separator,
                         header=None)
    df = df.replace(data.missing_value_token, np.nan)
    return df


def load_splits(data) -> List:
    """
    Loads train, validate, test splits from a directory.
    The data's missing values, which are represented in the data by
    missing_value_token, are replaced with np.nan.

    Argument Keywords:
    data_path -- path where subdirectories with splits are located
    data_title -- name of the dataset and it's associated splits
    missing_value_token -- specifies how missing values are represented
    in the dataset

    Returns a list of df_train, df_validate, df_test and, if loading splits
    of a cleaning dataset, df_validate_clean
    """
    df_train = pd.read_csv(data.splits_path+'train/' +
                           data.title+'_train.csv', header=None)
    df_validate = pd.read_csv(
        data.splits_path+'validate/'+data.title+'_validate.csv', header=None)
    df_test = pd.read_csv(data.splits_path+'test/' +
                          data.title+'_test.csv', header=None)
    dfs = [df_train, df_validate, df_test]

    if data.cleaning:
        df_validate_clean = pd.read_csv(data.splits_path+'validate_clean/' +
                                        data.title+'_validate_clean.csv',
                                        header=None)
        dfs.append(df_validate_clean)

    dfs = [df.replace(data.missing_value_token, np.nan) for df in dfs]
    return dfs


def random_dependency_generator(columns, n=10):
    """ Generates n random dependencies from a list of columns.

    Returns a dictionary with rhs's as keys and associated with it lhs
    combinations, representing a total of <= n dependencies.
    Note that the returned lhs-combinations are neither randomly
    distributed, nor are there exactly n dependencies returned.
    The way rand_length is generated makes it more likely for short
    lhs-combinations to be generated over long lhs-combinations.

    Keyword attributes:
    columns -- list of columns-names
    n -- int, indicating how many dependencies shall be returned
    """
    dependencies_dict = {}

    for _ in range(0, n):
        # at least 2 columns are necessary to form a dependency
        rand_length = random.randint(2, len(columns))
        lhs = random.sample(columns, rand_length)
        rhs = lhs.pop()
        lhs.sort()

        if rhs in dependencies_dict:
            if lhs not in dependencies_dict[rhs]:
                dependencies_dict[rhs].append(lhs)
        else:
            dependencies_dict[rhs] = [lhs]

    return dependencies_dict


def read_fds(fd_path):
    """  Returns a dictionary with FDs

    Functionally determined column serve as keys and arrays of functionally
    determining column-combinations as values.

    Keyword arguments:
    fd_path -- file-path to a metanome result of a FD detection algorithm
    """
    import re
    fd_dict = {}
    save_fds = False
    with open(fd_path) as f:
        for line in f:
            if save_fds:
                line = re.sub('\n', '', line)
                splits = line.split("->")

                # Convert to int and substract 1 to
                # start indexing of columns at 0
                lhs = [(int(x)-1) for x in splits[0].split(',')]
                rhs = int(splits[1])-1

                if rhs in fd_dict:
                    fd_dict[rhs].append(lhs)
                else:
                    fd_dict[rhs] = [lhs]

            if line == '# RESULTS\n':  # Start saving FDs
                save_fds = True

    return fd_dict


def get_performance(df_imputed: pd.DataFrame,
                    rhs: str,
                    lhs: list,
                    continuous_cols: list):
    """
    Create a dictionary containing metrics to measure the perfor-
    mance of a classifier. If the classified column contains continuous
    values, return a dictionary with keys {'nans', 'lhs', 'mse'}.
    'mse' is the mean squared error. If the classified column contains
    discrete values, return a dictionary with keys {'lhs', 'f1', 'recall',
    'precision'}.

    Keyword arguments:
    df_imputed -- dataframe. Column names are expected to be strings.
        One column needs to be called rhs, another one rhs+'_imputed'.
    rhs -- string, name of the column that has been imputed
    lhs -- list of strings, lhs of the FD
    continuous_cols -- list of strings, names of columns containing
    continuous values
    """
    from sklearn import metrics

    # turn everything into strings to facilitate df selection
    rhs = str(rhs)
    lhs = list(map(str, lhs))
    df_imputed.columns = list(map(str, df_imputed.columns))
    continuous_cols = list(map(str, continuous_cols))

    if rhs in continuous_cols:  # ignore NaNs, count them
        result_selector = df_imputed.loc[:, rhs+'_imputed'].isna()
        y_true = df_imputed.loc[~result_selector, rhs]
        y_pred = df_imputed.loc[~result_selector, rhs+'_imputed']
        no_nans = result_selector.sum()
    else:  # include NaNs, adjust dtype
        if isinstance(df_imputed.loc[0, rhs], str):
            df_imputed = df_imputed.fillna('no value')
        else:
            df_imputed = df_imputed.fillna(123456789)
        y_true = df_imputed.loc[:, rhs]
        y_pred = df_imputed.loc[:, rhs+'_imputed']

    if rhs in continuous_cols:
        mse = ''
        if len(y_pred) > 0:
            mse = metrics.mean_squared_error(y_true, y_pred)

        result = {'nans': no_nans, 'lhs': lhs, 'mse': mse}
    else:
        result = {
            'lhs': lhs,
            'f1': metrics.f1_score(y_true, y_pred, average='weighted'),
            'recall': metrics.recall_score(y_true,
                                           y_pred,
                                           average='weighted'),
            'precision': metrics.precision_score(y_true,
                                                 y_pred,
                                                 average='weighted')
        }
    return result


def df_to_ag_style(df: pd.DataFrame) -> TabularPredictor.Dataset:
    """
    Define a standardised way of passing DataFrames to AutoGluon by first
    casting the DataFrame to TabularPredictor.Dataset, then overwriting
    column-names with the string of the column's index.
    """
    ag_df = TabularPredictor.Dataset(df)
    ag_df.columns = [str(i) for i in df.columns]
    return ag_df


def map_index(x: Any, column_map: dict) -> str:
    """Makes column list index human-readable."""
    return f'{column_map[x]}'
