def read_fds(fd_path):
    """  Returns a dictionary with FDs

    Functionally determined column serve as keys and arrays of functionally
    determining column-combinations as values.

    Keyword arguments:
    fd_path -- file-path to a metanome result of a FD detection algorithm
    """
    import re
    fd_dict = dict()
    save_fds = False
    with open(fd_path) as f:
        for line in f:
            if save_fds:
                line = re.sub('\n', '', line)
                splits = line.split("->")

                # Convert to int and substract 1 to
                # start indexing of columns at 0
                splits[0] = [(int(x)-1) for x in splits[0].split(',')]
                splits[1] = int(splits[1])-1

                if splits[1] in fd_dict:
                    fd_dict[splits[1]].append(splits[0])
                else:
                    fd_dict[splits[1]] = [splits[0]]

            if line == '# RESULTS\n':  # Start saving FDs
                save_fds = True

    return fd_dict


def fd_imputer(df_test, df_train, fd):
    """ Imputes a column of a dataframe using a FD.
    Returns a dataframe with an additional column named
    rhs+'_imputed'. Only returns rows for which an imputation has been
    successfully performed. The returned dataframe's index is the one
    used by df_test.

    Keyword arguments:
    df_test -- dataframe where a column shall be imputed
    df_train -- dataframe on which fds were detected
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
                          df_test.iloc[:, relevant_cols].reset_index(),
                          on=lhs,
                          suffixes=('_imputed', '')).set_index('index')

    df_test_imputed = pd.merge(df_test,
                               df_imputed.loc[:, str(rhs)+'_imputed'],
                               left_index=True,
                               right_index=True,
                               how='left')

    return df_test_imputed


def ml_imputer(df_train, df_test, impute_column, overfit=False):
    """ Imputes a column using DataWigs SimpleImputer

    Keyword arguments:
    df_train -- dataframe containing the train set
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

    imputer = SimpleImputer(
        input_columns=input_columns,
        output_column=impute_column,
        output_path='imputer_model/'
    )

    if overfit:
        imputer.fit(train_df=df_test, num_epochs=10, patience=3)
    else:
        imputer.fit(train_df=df_train, num_epochs=10, patience=3)

    predictions = imputer.predict(df_test)
    return predictions


def save_df_split(data_title, df, splits_path, split_ratio):
    """ Splits and saves a dataframe to the harddrive.

    Keyword arguments:
    data_title -- a string naming the data
    df -- a pandas data frame
    splits_path -- file-path to folder where splits should be saved
    split_ratios -- list of three floats that are in sum <= 1,
    train/test/validate set ratio, e.g. [0.8, 0.1, 0.1]
    metanome_data_path -- path to folder where metanome reads its data
    """
    import os
    from datawig.utils import random_split

    ratio_train, ratio_validate, ratio_test = (split_ratio)

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

    try:
        df.to_csv(splits_path+data_title+'.csv', header=None)
        print('Dataset successfully written to '+splits_path+data_title+'.csv')
    except TypeError:
        print('Could not save dataframe to '+splits_path+data_title)

    for key in splits:
        try:
            splits[key]['df'].to_csv(
                splits[key]['path']+data_title+'_'+key+'.csv', header=None)
            print(key+' set successfully written to ' +
                  splits[key]['path']+data_title+'_'+key+'.csv')
        except TypeError:
            print("Something went wrong writing the splits.")
