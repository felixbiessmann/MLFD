import pandas as pd
from lib.helpers import get_performance
from autogluon.tabular import TabularPredictor as task


def ml_imputer(df_train, df_validate, df_test, label_column: str):
    train_data = task.Dataset(df_train)
    test_data = task.Dataset(df_test)
    validate_data = task.Dataset(df_validate)
    train_data.columns = [str(i) for i in df_train.columns]
    test_data.columns = [str(i) for i in df_test.columns]
    validate_data.columns = [str(i) for i in df_validate.columns]

    d = 'agModels-predictClass'  # folder to store trained models
    predictor = task(label=str(label_column),
                     path=d).fit(train_data=train_data, tuning_data=test_data)

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
