def load_dataframes(splits_path, data_title, missing_value_token):
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

    return (dfs)


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
    rhs = list(fd)[0]  # select the fd right hand side
    lhs = fd[rhs]  # select the fd left hand side
    relevant_cols = lhs.copy()
    relevant_cols.append(rhs)

    """ By dropping duplicates in the train set before merging dataframes
    memory usage is heavily reduced"""
    df_train_reduced = df_train.iloc[:, relevant_cols].drop_duplicates(
        subset=relevant_cols,
        keep='first',
        inplace=False)

    """ reset_index() and set_index() is a hacky way to save df_test's index
    which is normally lost due to pd.merge(). """
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


def run_fd_imputer_on_fd_set(df_train, df_validate, fds, continuous_cols=[]):
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

    Returns:
    A dictionary consisting of rhs's as keys with associated performance
    metrics for each lhs. Performance metrics are precision, recall and
    f1-measure for classifiable data and the mean squared error as well as
    the number of unsucessful imputations for continuous data.
    """
    from sklearn import metrics
    fd_imputer_results = {}

    for rhs in fds:
        rhs_results = []

        for lhs in fds[rhs]:
            fd = {rhs: lhs}
            print(fd)

            df_fd_imputed = fd_imputer(df_train, df_validate, fd)

            # make sure that value for missing data is of same type as
            # row to be imputed to avoid mix of labels and continuous numbers
            if isinstance(df_fd_imputed.iloc[0, rhs], str):
                df_fd_imputed = df_fd_imputed.fillna('no value')
                y_pred = df_fd_imputed.loc[:, str(rhs)+'_imputed']
                y_true = df_fd_imputed.loc[:, rhs]
            else:
                df_imputed_col = df_fd_imputed.loc[:, str(rhs)+'_imputed']
                nan_result_selector = df_imputed_col.isna()
                nans = nan_result_selector.sum()

                # ignore nans to compute MSE
                y_pred = df_fd_imputed.loc[~nan_result_selector,
                                           str(rhs)+'_imputed']
                y_true = df_fd_imputed.loc[~nan_result_selector,
                                           rhs]

            if rhs in continuous_cols:
                mse = ''

                if len(y_pred) > 0:
                    mse = metrics.mean_squared_error(y_true, y_pred)

                result = {'nans': nans, 'lhs': lhs, 'mse': mse}
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
            rhs_results.append(result)
        fd_imputer_results[rhs] = rhs_results

    return fd_imputer_results


def ml_imputer(df_train, df_validate, df_test, impute_column):
    """ Imputes a column using DataWigs SimpleImputer

    Keyword arguments:
    df_train -- dataframe containing the train set
    df_validate -- dataframe containing the validation dataset
    df_test -- dataframe containing the test set
    impute_column -- position (int) of column to be imputed, starting at 0
    """
    from datawig import SimpleImputer

    columns = list(df_train.columns)

    # SimpleImputer expects dataframes to have headers.
    impute_column = str(impute_column)
    input_columns = [str(col) for col in columns if col != impute_column]
    df_train.columns = [str(i) for i in df_train.columns]
    df_test.columns = [str(i) for i in df_test.columns]
    df_validate.columns = [str(i) for i in df_validate.columns]

    imputer = SimpleImputer(
        input_columns=input_columns,
        output_column=impute_column,
        output_path='imputer_model/'
    )

    imputer.fit(train_df=df_train,
                test_df=df_validate,
                num_epochs=10,
                patience=3)

    predictions = imputer.predict(df_test)
    return predictions


def run_ml_imputer_on_fd_set(df_train, df_validate, df_test, fds,
                             continuous_cols=[]):
    """ This needs documentation
    """
    from sklearn import metrics
    ml_imputer_results = {}
    for rhs in fds:
        rhs_results = []
        for lhs in fds[rhs]:
            relevant_cols = lhs + [rhs]

            # make sure that datawig doesn't perform regression on categories
            # also, select relevant subsets
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
            df_imputed = fd_imputer.ml_imputer(df_subset_train,
                                               df_subset_validate,
                                               df_subset_test,
                                               str(rhs))

            y_pred = df_imputed.loc[:, str(rhs)+'_imputed']
            y_true = df_imputed.loc[:, str(rhs)]

            if rhs in continuous_cols:
                result = {
                    'lhs': lhs,
                    'mse': metrics.mean_squared_error(y_true, y_pred)
                }
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
            rhs_results.append(result)
        ml_imputer_results[rhs] = rhs_results
    return ml_imputer_results


def index_as_first_column(df):
    df = df.reset_index()
    df.columns = [x for x in range(0, len(df.columns))]
    return df


def split_df(data_title, df, split_ratio, splits_path=''):
    """ Splits a dataframe into train-, validate- and test-subsets.
    If a splits_path is provided, splits will be saved to the harddrive.

    Returns a tuple (df_train, df_validate, df_test) if the splits_path
    is an empty string.

    Keyword arguments:
    data_title -- a string naming the data
    df -- a pandas data frame
    splits_path -- file-path to folder where splits should be saved
    split_ratio -- train/test/validate set ratio, e.g. [0.8, 0.1, 0.1]
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

    if splits_path != '':
        try:
            df.to_csv(splits_path+data_title+'.csv', header=None, sep=',')
            print('Dataset successfully written to '
                  + splits_path+data_title +
                  '.csv')
        except TypeError:
            print('Could not save dataframe to '+splits_path+data_title)

        for key in splits:
            try:
                splits[key]['df'].to_csv(
                    splits[key]['path']+data_title+'_'+key+'.csv',
                    index=False, header=False)
                print(key+' set successfully written to ' +
                      splits[key]['path']+data_title+'_'+key+'.csv')
            except TypeError:
                print("Something went wrong writing the splits.")

    return(splits['train']['df'],
           splits['validate']['df'],
           splits['test']['df'])
