def check_split_for_duplicates(list_of_dfs):
    """ Prints and returns number of duplicate entries in a list of dfs """
    import pandas as pd
    no_duplicates = sum(pd.concat(list_of_dfs).duplicated())
    print(str(no_duplicates) + ' duplicates detected in the splittings.')
    return no_duplicates


def load_dataframes(splits_path, data_title, missing_value_token):
    """ Loads train, validate, test splits from a directory.
    The data's missing values, which are represented in the data by
    missing_value_token, are deleted.

    Argument Keywords:
    splits_path -- path where subdirectories with splits are located
    data_title -- name of the dataset and it's associated splits
    missing_value_token -- specifies how missing values are represented
    in the dataset
    """

    import pandas as pd
    import numpy as np
    df_train = pd.read_csv(splits_path+'train/' +
                           data_title+'_train.csv', header=None)
    df_validate = pd.read_csv(
        splits_path+'validate/'+data_title+'_validate.csv', header=None)
    df_test = pd.read_csv(splits_path+'test/' +
                          data_title+'_test.csv', header=None)

    dfs = [df_train, df_validate, df_test]
    dfs = [df.replace(missing_value_token, np.nan) for df in dfs]
    dfs = [df.dropna(axis=0) for df in dfs]  # drop rows with nans

    return (dfs)


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
    import random
    dependencies_dict = {}

    for i in range(0, n):
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


def fd_imputer(df_train, df_validate, fd):
    """ Imputes a column of a dataframe using a FD.
    Returns a dataframe with an additional column named
    rhs+'_imputed'. Only returns rows for which an imputation has been
    successfully performed. The returned dataframe's index is the one
    used by df_validate.

    Keyword arguments:
        df_train -- dataframe on which fds were detected
        df_validate -- dataframe on which a column shall be imputed
        fd -- dictionary containing the RHS as key and a list of LHS as value
    """
    import pandas as pd
    rhs = list(fd)[0]
    lhs = fd[rhs]
    relevant_cols = lhs.copy()
    relevant_cols.append(rhs)

    """ By dropping duplicates in the train set before merging dataframes
    memory usage is heavily reduced"""
    df_train_reduced = df_train.iloc[:, relevant_cols].drop_duplicates(
        subset=relevant_cols,
        keep='first',
        inplace=False)

    """ reset_index() and set_index() is a way to preserver df_test's index
    which is normally lost due to pd.merge()."""
    df_imputed = pd.merge(df_train_reduced,
                          df_validate.iloc[:, relevant_cols].reset_index(),
                          on=lhs,
                          suffixes=('_imputed', '')).set_index('index')

    df_validate_imputed = pd.merge(df_validate,
                                   df_imputed.loc[:, str(rhs)+'_imputed'],
                                   left_index=True,
                                   right_index=True,
                                   how='left')

    return df_validate_imputed


def run_fd_imputer_on_fd_set(df_train, df_validate, fds,
                             continuous_cols=[], debug=False):
    """ Runs fd_imputer for every fd contained in a dictionary on a split df.

    Executes fd_imputer() for every fd in fds. df_train and df_split should be
    created using split_df(). continuous_cols is used to distinguish between
    columns containing continuous_cols numbers and classifiable objects.

    Keyword arguments:
        df_train -- a df on which fds have been detected. Needs to have the
        same origin df as df_validate
        df_validate -- a df on which no fds have been detected. Needs to have
        the same origin df as df_train
        fds -- a dictionary with an fd's possible rhs as values and a list of
        lhs-combinations as value of fds[rhs].
        debug -- boolean. If True, return a list of imputed dataframes instead
        of dictionary with metrics

    Returns:
    A dictionary consisting of rhs's as keys with associated performance
    metrics for each lhs. Performance metrics are precision, recall and
    f1-measure for classifiable data and the mean squared error as well as
    the number of unsucessful imputations for continuous data.
    """
    fd_imputer_results = {}

    if debug:
        list_of_imputed_dfs = []
    for rhs in fds:
        rhs_results = []

        for lhs in fds[rhs]:
            fd = {rhs: lhs}
            print(fd)

            df_fd_imputed = fd_imputer(df_train, df_validate, fd)
            result = get_performance(df_fd_imputed, str(rhs), lhs,
                                     continuous_cols)
            rhs_results.append(result)
            if debug:
                list_of_imputed_dfs.append(df_fd_imputed)

        fd_imputer_results[rhs] = rhs_results

    if debug:
        return list_of_imputed_dfs
    else:
        return fd_imputer_results


def get_performance(df_imputed, rhs: str, lhs: list, continuous_cols: list):
    """ Create a dictionary containing metrics to measure the perfor-
    mance of a classifier. If the classified column contains continuous
    values, return a dictionary with keys {'nans', 'lhs', 'mse'}.
    'mse' is the mean squared error. If the classified column contains
    discrete values, return a dictionary with keys {'lhs', 'f1', 'recall',
    'precision'}.

    Keyword arguments:
    df_imputer -- dataframe. Column names are expected to be strings.
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
            'f1': metrics.f1_score(y_true,
                                   y_pred,
                                   average='weighted'),
            'recall': metrics.recall_score(y_true,
                                           y_pred,
                                           average='weighted'),
            'precision': metrics.precision_score(y_true,
                                                 y_pred,
                                                 average='weighted')
        }
    return result


def ml_imputer(df_train, df_validate, df_test, impute_column, epochs=10):
    """ Imputes a column using DataWigs SimpleImputer

    Keyword arguments:
    df_train - - dataframe containing the train set
    df_validate - - dataframe containing the validation dataset
    df_test - - dataframe containing the test set
    impute_column - - position(int) of column to be imputed, starting at 0
    epochs - - int, number of training_epochs to use with SimpleImputer.fit
    """
    from datawig import SimpleImputer
    import tempfile

    columns = list(df_train.columns)

    # SimpleImputer expects dataframes to have stringtype headers.
    impute_column = str(impute_column)

    input_columns = [str(col) for col in columns if str(col) != impute_column]

    df_train.columns = [str(i) for i in df_train.columns]
    df_test.columns = [str(i) for i in df_test.columns]
    df_validate.columns = [str(i) for i in df_validate.columns]

    with tempfile.TemporaryDirectory() as tmpdirname:
        imputer = SimpleImputer(
            input_columns=input_columns,
            output_column=impute_column,
            output_path=tmpdirname
        )

    imputer.fit(train_df=df_train,
                test_df=df_validate,
                num_epochs=epochs,
                patience=3)

    predictions = imputer.predict(df_test)
    return (predictions)


def run_ml_imputer_on_fd_set(df_train, df_validate, df_test, fds,
                             continuous_cols=[], cycles=10):

    """ Runs ml_imputer for every fd contained in a dictionary on a split df.

    Executes fd_imputer() for every fd in fds. df_train and df_split should be
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

            # cast rhs to str to make sure that datawig doesn't perform
            # regression on categories. Also, select relevant subsets to
            # improve performance.
            if rhs not in continuous_cols:
                df_subset_train = df_train.iloc[:, relevant_cols].astype(
                    {rhs: str})
                df_subset_validate = df_validate.iloc[:, relevant_cols].astype(
                    {rhs: str})
                df_subset_test = df_test.iloc[:, relevant_cols].astype(
                    {rhs: str})
            else:
                df_subset_train = df_train.iloc[:, relevant_cols]
                df_subset_validate = df_validate.iloc[:, relevant_cols]
                df_subset_test = df_test.iloc[:, relevant_cols]

            print(lhs, rhs)
            df_imputed = ml_imputer(df_subset_train,
                                    df_subset_validate,
                                    df_subset_test,
                                    str(rhs),
                                    cycles)

            result = get_performance(df_imputed, rhs, lhs, continuous_cols)
            rhs_results.append(result)
        ml_imputer_results[rhs] = rhs_results
    return ml_imputer_results


def index_as_first_column(df):
    df = df.reset_index()
    df.columns = [x for x in range(0, len(df.columns))]
    return df


def split_df(data_title, df, split_ratio, splits_path=''):
    """ Splits a dataframe into train-, validate - and test-subsets.
    If a splits_path is provided, splits will be saved to the harddrive.

    Returns a tuple(df_train, df_validate, df_test) if the splits_path
    is an empty string.

    Keyword arguments:
    data_title - - a string naming the data
    df - - a pandas data frame
    splits_path - - file-path to folder where splits should be saved
    split_ratio - - train/test/validate set ratio, e.g. [0.8, 0.1, 0.1]
    """
    import os
    from datawig.utils import random_split
    import numpy as np

    ratio_train, ratio_validate, ratio_test = (split_ratio)

    # compare first col with index. if not equal...
    if not np.array_equal(df.iloc[:, 0].values, np.array(df.index)):
        # ...set index as 0th column such that fd_imputer can use it to impute
        print('kein doppelter index')
        df = index_as_first_column(df)

    splits = {
        'train': {'path': splits_path+'train/',
                  'ratio': ratio_train},
        'validate': {'path': splits_path+'validate/',
                     'ratio': ratio_validate},
        'test': {'path': splits_path+'test/',
                 'ratio': ratio_test}
    }

    for key in splits:
        if not os.path.exists(splits[key]['path']):
            os.mkdir(splits[key]['path'])

    ratio_rest = splits['validate']['ratio'] + splits['test']['ratio']

    ratios = [splits['train']['ratio'], ratio_rest]
    splits['train']['df'], rest = random_split(df, split_ratios=ratios)

    # calculate ratio_validate and ratio_test relative to ratio_rest
    splits['test']['rel_ratio'] = splits['test']['ratio'] / ratio_rest
    splits['validate']['rel_ratio'] = 1 - splits['test']['rel_ratio']
    rel_ratios = [splits['test']['rel_ratio'],
                  splits['validate']['rel_ratio']]

    splits['validate']['df'], splits['test']['df'] = random_split(
        rest, split_ratios=rel_ratios)

    if splits_path == '':
        return(splits['train']['df'],
               splits['validate']['df'],
               splits['test']['df'])
    else:
        try:
            df.to_csv(splits_path+data_title+'.csv', header=None, sep=',',
                      index=False)
            print('Dataset successfully written to '
                  + splits_path+data_title +
                  '.csv')
        except TypeError:
            print('Could not save dataframe to '+splits_path+data_title)

        for key in splits:
            try:
                splits[key]['df'].to_csv(
                    splits[key]['path']+data_title+'_'+key+'.csv', sep=',',
                    index=False, header=None)
                print(key+' set successfully written to ' +
                      splits[key]['path']+data_title+'_'+key+'.csv')
            except TypeError:
                print("Something went wrong writing the splits.")
