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
    rhs+'_imputed'.

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

    """ reset_index() and set_index() is a hacky way to save df_test's index
    which is normally lost due to pd.merge(). """
    df_imputed = pd.merge(df_train.iloc[:, relevant_cols],
                          df_test.iloc[:, :].reset_index(),
                          on=lhs,
                          suffixes=('_imputed', '')).set_index('index')
    return df_imputed


def ml_imputer(df_train, df_test, impute_column):
    """ Imputes a column using DataWigs SimpleImputer

    Keyword arguments:
    df_train -- dataframe containing the train set
    df_test -- dataframe containing the test set
    impute_column -- position (int) of column to be imputed, starting at 0
    """
    from datawig import SimpleImputer
    # from sklearn.metrics import f1_score

    columns = list(df_train.columns)

    # SimpleImputer expects dataframes to have headers
    impute_column = str(impute_column)
    input_columns = [str(col) for col in columns if col != impute_column]
    df_train.columns = [str(i) for i in range(0, len(df_train.columns))]
    df_test.columns = [str(i) for i in range(0, len(df_test.columns))]

    imputer = SimpleImputer(
        input_columns=input_columns,
        output_column=impute_column,
        output_path='imputer_model/'
    )

    imputer.fit(train_df=df_train)
    predictions = imputer.predict(df_test)
    '''f1 = f1_score(predictions[impute_column], predictions[impute_column+'_imputed'].astype(int),
    average='weighted')
    print(f1)'''
    return predictions


def save_df_split(data_title, df, splits_path, metanome_data_path,
                  split_ratio):
    """ Splits and saves a dataframe to the harddrive.

    Keyword arguments:
    data_title -- a string naming the data
    df -- a pandas data frame
    splits_path -- file-path to folder where splits should be saved
    split_ratios -- list of two floats <= 1, train/test set ratio,
    e.g. [0.8, 0.2]
    metanome_data_path -- path to folder where metanome reads its data
    """
    import os
    from datawig.utils import random_split

    train_path = splits_path+'train/'
    test_path = splits_path+'test/'

    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    df_train, df_test = random_split(df, split_ratios=split_ratio)

    try:
        df.to_csv(splits_path+data_title+'.csv', header=None)
        print('Dataset successfully written to '+splits_path+data_title)
        df_train.to_csv(train_path+data_title+'_train.csv', header=None)
        print('Train set successfully written to ' +
              train_path+data_title+'_train')
        try:
            os.system('cp '+train_path+data_title +
                      '_train.csv '+metanome_data_path)
            print('Copied successfully train-dataset to '+metanome_data_path)
        except SyntaxError:
            print('Could not copy train-set to metanome data path.')
        df_test.to_csv(test_path+data_title+'_test.csv', header=None)
        print('Test set successfully written to '+test_path+data_title+'_test')
    except TypeError:
        print("Something went wrong writing the splits.")
