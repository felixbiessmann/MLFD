import autogluon.core as ag
import timeit
import datetime
import random
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
import lib.helpers as helps
import lib.optimizer as opt
import lib.constants as c
import lib.imputer as imp


def clean_data(data: c.Dataset, save=True, *args, **kwargs):
    """
    Train a model on the dirty (default) train-test split. Then, calculate
    cleaning performance by comparing predicted labels on the dirty dataset
    to the clean labels from the clean dataset.

    Depending on the type of comparison between the validate sets, this is
    either an error-detection experiment, or a cleaning experiment.
    """

    gbm_options = {  # non-default hyperparameters for lightGBM
                     # gradient boosted trees
                    'num_boost_round': 10,  # number of boosting rounds (controls training time of GBM models)
                    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
    }

    config = {"random_state": 0,
              "verbosity": 2,
              "precision_threshold": 0.01,
              "numerical_confidence_quantile": 0.7,
              "force_multiclass": True,  # prevents regression from happening at all
              "time_limit": 500,  # how long autogluon trains
              "replace_nans": False,  # replace values that weren't imputed with NaNs
              "force_retrain": True,  # skip loading ag models by force
              "train_cleaning_cols": True,  # train only on cols that contain errors
              "n_rows": 1000,
              "label_count_threshold": 1,
              "hyperparameters": {
                  'GBM': gbm_options,
                },
              'hyperparameter_tune_kwargs': {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
                'num_trials': 5,  # try at most 5 different hyperparameter configurations for each type of model
                'scheduler': 'local',
                'searcher': 'auto',  # to tune hyperparameters using Bayesian optimization routine with a local scheduler
                }
              }

    logger = logging.getLogger('pfd')
    logger.debug(f"Start cleaning experiment with dataset {data.title}.")
    logger.debug("For each column, a model for cleaning will be trained on "
                 "all other columns.")
    if not data.cleaning:
        logger.info(f"The dataset {data.title} is not suitable for a cleaning "
                    "experiment. Please choose a dataset that has a clean "
                    "and a dirty version of the data available.")
        return False

    original_df_clean = helps.load_original_data(data, load_dirty=False)
    original_df_dirty = helps.load_original_data(data, load_dirty=True)

    original_df_clean = original_df_clean.iloc[:config['n_rows'], :]
    original_df_dirty = original_df_dirty.iloc[:config['n_rows'], :]

    cols = data.column_map.keys()
    if config['train_cleaning_cols']:
        cols = data.cols_with_errors

    result = []
    result.append(config)
    global_pred_y = np.array([])
    global_clean_y = original_df_clean.iloc[:, cols].to_numpy().T.flatten()
    global_dirty_y = original_df_dirty.iloc[:, cols].to_numpy().T.flatten()

    for label in cols:

        # cast target to str to avoid dtype-issues when cleaning
        df_clean = original_df_clean.copy()
        df_clean[label] = df_clean[label].astype('str')

        df_dirty = original_df_dirty.copy()
        df_dirty[label] = df_dirty[label].astype('str')

        r = {'label': label}
        logger.info('\n~~~~~')
        logger.info(f'Investigating RHS {data.column_map[label]} ({label})')

        imputer, r['model_checksum'] = imp.train_cleaning_model(df_dirty,
                                                                label,
                                                                **config)
        logger.info("Trained global imputer with complete LHS.")

        logger.debug("Successfully trained the model.")

        # Cast to string to make sure that comparisons with imputed values
        # work as intended.
        df_dirty_y_true = df_dirty.loc[:, label].astype(str)
        df_clean_y_true = df_clean.loc[:, label].astype(str)

        logger.debug("Predicting values.")
        probas = imputer.predict(df_dirty,
                                 precision_threshold=config['precision_threshold'],
                                 return_probas=True)

        se_predicted = imp.make_cleaning_prediction(df_dirty,
            probas,
            label)

        se_predicted = se_predicted.astype(str)

        # count erorrs in the dirty dataset
        error_positions = df_clean_y_true.fillna('') != df_dirty_y_true.fillna('')

        n_errors = sum(error_positions)
        tp = sum(df_clean[label][error_positions].astype(str) == se_predicted[error_positions])
        r['n_errors_in_dirty'] = n_errors

        if config['replace_nans']:
            se_predicted[pd.isna(se_predicted)] = df_dirty_y_true[pd.isna(se_predicted)]

        global_pred_y = np.append(global_pred_y, se_predicted)
        logger.debug("Successfully predicted values.")

        logger.debug('Measuring cleaning-performance.')
        r['error_cleaning'] = helps.cleaning_performance(df_clean_y_true,
                                                     se_predicted,
                                                     df_dirty_y_true)
        logger.debug('Measuring error-detection performance.')
        r['error_detection'] = helps.error_detection_performance(df_clean_y_true,
                                                             se_predicted,
                                                             df_dirty_y_true)

        logger.info("Calculated a cleaning performance of "
                    f"f1-score {round(r['error_cleaning'], 5)}.")
        logger.info("Calculated a error detection performance "
                    f"of f1-score {round(r['error_detection'], 5)}.")
        result.append(r)

    global_detection = helps.error_detection_performance(global_clean_y,
                                                     pd.Series(global_pred_y),
                                                     global_dirty_y)
    global_cleaning = helps.cleaning_performance(pd.Series(global_clean_y),
                                             pd.Series(global_pred_y),
                                             pd.Series(global_dirty_y))
    logger.info('Global error detection performance of F1-Score '
                f'{global_detection}.')
    logger.info('Global error cleaning performance of F1-Score '
                f'{global_cleaning}.')
    result.append({'global_error_detection': global_detection,
                   'global_error_cleaning': global_cleaning})

    # always save results
    now = round(datetime.datetime.now().timestamp())
    result.append({'run_at_timestamp': now})
    path = f"{data.results_path}{now}_clean_data.p"
    helps.save_pickle(result, path)

    logger.info('\n\n~~~~~~~~~~~~~~~~~~~~\n')
    logger.info(result)


def global_predictor_explained(data: c.Dataset,
                               df_train: pd.DataFrame,
                               df_validate: pd.DataFrame,
                               df_test: pd.DataFrame
                               ) -> Tuple[pd.DataFrame, float, str, int]:
    """
    Asks the user for a RHS and trains a global predictor for a given dataset.
    The predictor is then explains it using feature_permutation.

    Returns a Tuple with the Feature-Importance-Dataframe, the
    performance-dict, the metric the predictor is evaluated on and the index
    of the RHS column.
    """
    logger = logging.getLogger('pfd')
    print("Compute a PFD.")
    print("What is the index of the RHS to investigate?")
    print(repr(data.column_map))
    rhs = int(input(''))  # select column based on integers
    logger.debug(f'User chose rhs {rhs}')

    logger.info("Begin global predictor training with complete LHS.")
    df_imp, measured_performance, metric = opt.get_importance_pfd(df_train,
                                                                  df_validate,
                                                                  df_test,
                                                                  rhs)
    # add column-description to importance-dataframe
    df_imp['description'] = df_imp.index.to_series().apply(
        lambda x: data.column_map.get(x, 'NA'))

    return (df_imp, measured_performance[metric], metric, rhs)


def manual_pfd(data, *args, **kwargs):
    """
    Use feature permutation to compute the PFD of a dataset. The user is
    prompted to insert how much mean absolute deviation from the mean
    function value they want to secrifice for a smaller LHS.
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start manual search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data)
    df_imp, measured_performance, metric, rhs = global_predictor_explained(
        data, df_train, df_validate, df_test)

    exclude_cols = []
    include_cols = list(df_train.columns)
    lhs = [c for c in include_cols if c != rhs]

    while True:
        logger.info(f"Trained a predictor with {metric} "
                    f"{measured_performance}.")
        print("These are the feature importances found via "
              "feature permutation:")
        print(df_imp.loc[lhs, ['description', 'importance']].sort_values(
            'importance', ascending=False))
        print(f'Excluded: {exclude_cols}')

        print("Which columns do you want to exclude? (q to quit)")
        i = input('')
        if i == 'q':
            break

        exclude_cols = [int(c) for c in i.split(',')]
        include_cols = [c for c in include_cols if c not in exclude_cols]
        lhs = [c for c in include_cols if c != rhs]

        logger.info(f"Begin predictor training with LHS {lhs}")
        measured_performance = opt.iterate_pfd(include_cols,
                                               df_train,
                                               df_validate,
                                               df_test,
                                               str(rhs))


def linear_pfd(data, *args, **kwargs):
    """
    Uses feature permutation to automatically search for minimal PFDs.

    In AutoGluon, higher performance metrics are always better. This leads to
    the circumstnace that the MSE is negative! So don't dispair!
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start linear search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data)

    df_imp, measured_performance, metric, rhs = global_predictor_explained(
        data, df_train, df_validate, df_test)

    include_cols = list(df_train.columns)
    lhs = [c for c in include_cols if c != rhs]

    logger.info(f"Trained a predictor with {metric} "
                f"{measured_performance}.")
    print("These are the feature importances found via "
          "feature permutation:")
    print(df_imp.loc[lhs, ['description', 'importance']].sort_values(
        'importance', ascending=False))
    print(f"What's your threshold for {metric}?")
    threshold = float(input(''))

    for i in range(len(df_train.columns)):
        if measured_performance < threshold:
            logger.info("The newly trained model's performance of "
                        f"{measured_performance} is below the threshold of "
                        f"{threshold}. Stopping the search.")
            break

        # the i-least important column
        exclude_col = int(df_imp.iloc[-(i+1), :].name)
        include_cols = [c for c in include_cols if c != exclude_col]
        lhs = [c for c in include_cols if c != rhs]

        logger.info(f"Begin predictor training with LHS {lhs}")
        measured_performance = opt.iterate_pfd(include_cols,
                                               df_train,
                                               df_validate,
                                               df_test,
                                               rhs)
        logger.info(
            f"Trained a predictor with {metric} {measured_performance}")


def binary_pfd(data, *args, **kwargs):
    """
    Uses feature permutation to automatically search for minimal PFDs.

    In AutoGluon, higher performance metrics are always better. This leads to
    the circumstnace that the MSE is negative! So don't dispair!
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start linear search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data)

    df_imp, measured_performance, metric, rhs = global_predictor_explained(
        data, df_train, df_validate, df_test)

    include_cols = list(df_train.columns)
    lhs = [c for c in include_cols if c != rhs]

    logger.info(f"Trained a predictor with {metric} "
                f"{measured_performance}.")
    print("These are the feature importances found via "
          "feature permutation:")
    print(df_imp.loc[lhs, ['description', 'importance']].sort_values(
        'importance', ascending=False))
    print(f"What's your threshold for {metric}?")
    threshold = float(input(''))

    data = df_imp.loc[:, 'importance']
    iterate_pfd = opt.get_pfd_iterator(df_train, df_validate, df_test, rhs)
    opt.run_binary_search(data, threshold, iterate_pfd)


def jump_pfd(data, *args, **kwargs):
    """
    Uses refined searching stategies to greedily jump to a node in the
    search lattice.

    AG's feature_permutation returns feature importances that aren't
    additive and thus don't sum up to equal the model's loss function.
    Which is why I normalize the permutation feature importances to the
    model's measured_performance. This is not theoretically sound though
    -- there is no reason to think that FIs calculated for features
    [x_1, x_2, x_3, x_4] are the same FIs calculated for features
    [x_2, x_3, x_4] when using feature_permutation.

    Note that in AutoGluon, higher performance metrics are always better.
    This leads to the circumstance that the MSE is negative!

    Also, shap.TreeExplainer is capable of explaining a model's log_loss,
    which might be worthwhile investigating in future versions.
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start automatical search of PFDs for dataset {data.title}")

    df_train, df_validate, df_test = helps.load_splits(data)

    df_imp, measured_performance, metric, rhs = global_predictor_explained(
        data, df_train, df_validate, df_test)
    exclude_cols = []
    include_cols = list(df_train.columns)
    lhs = [c for c in include_cols if c != rhs]

    logger.info(f"Trained a predictor with {metric} "
                f"{measured_performance}.")
    print("These are the feature importances found via "
          "feature permutation:")
    print(df_imp.loc[lhs, ['description', 'importance']].sort_values(
        'importance', ascending=False))
    print(f"What's your threshold for {metric}?")
    threshold = float(input(''))

    logger.debug("User set threshold of {threshold}")

    # the margin is how much performance we can shave off
    margin = measured_performance - threshold
    if margin < 0:
        logger.info(f"The set threshold of {threshold} is below "
                    "the measured_performance of {measured_performance}. "
                    "Stopping the search.")

    if measured_performance < threshold:
        logger.info("The newly trained model's performance of "
                    f"{measured_performance} is below the threshold of "
                    f" {threshold}. Stopping the search.")

    df_imp['normalized_importance'] = (df_imp.loc[:, 'importance']
                                       / df_imp.loc[:, 'importance'].sum()) * measured_performance
    df_importance_cumsum = df_imp.sort_values(
        'normalized_importance', ascending=True).cumsum()
    importance_distance = df_importance_cumsum.loc[:,
                                                   'normalized_importance'] - margin

    logger.debug("Calculated the following importance distance "
                 f"{importance_distance}")
    exclude_cols = [int(x[0]) for x in importance_distance.iteritems()
                    if x[1] < 0]
    include_cols = [c for c in include_cols if c not in exclude_cols]
    lhs = [c for c in include_cols if c != rhs]

    logger.info("Training a predictor next to check threshold.")
    logger.info("Begin predictor training")
    measured_performance = opt.iterate_pfd(include_cols,
                                           df_train,
                                           df_validate,
                                           df_test,
                                           rhs)
    logger.info("Using Feature Permutation Importances, jumped "
                f"to LHS {lhs}, resulting in a Model with {metric} "
                f"{round(measured_performance, 3)}. The threshold aimed for "
                f"was a {metric} of {threshold}.")


def split_dataset(data, save=True):
    """
    Splits a dataset into train, validate and test subsets.
    When a dataset is a dataset used in the cleaning experiment,
    this function returns two validate-splits: the default one
    from the dirty data, and the _clean one from the clean data.

    Be cautious when using, this might overwrite existing data.
    """
    print(f'''
          You are about to split dataset {data.title}.
          If you continue, splits will be saved to {data.splits_path}.
          This might overwrite and nullify existing results.
          Do you want to proceed? [y/N]''')
    logger = logging.getLogger('pfd')
    sure = input('')
    if sure == 'y':
        split_ratio = (0.8, 0.1, 0.1)
        rndint = random.randint(0, 10000)
        df_clean = helps.load_original_data(data)
        save_dict = {}
        clean_train, clean_validate, clean_test = helps.split_df(df_clean,
                                                                 split_ratio,
                                                                 random_state=rndint)
        save_dict['train'] = clean_train
        save_dict['test'] = clean_test
        save_dict['validate'] = clean_validate
        logger.info('Splitting has been successful. '
                    'Original data duplicates: '
                    f'{str(sum(df_clean.duplicated()))}')
        if data.cleaning:
            logger.info('Selected dataset is a dataset used for a cleaning '
                        'experiment. Splitting dirty and clean data.')
            df_dirty = helps.load_original_data(data, load_dirty=True)
            dirty_train, dirty_validate, dirty_test = helps.split_df(df_dirty,
                                                                     split_ratio,
                                                                     random_state=rndint)
            save_dict['train'] = dirty_train
            save_dict['test'] = dirty_test
            save_dict['validate_clean'] = save_dict['validate']
            save_dict['validate'] = dirty_validate
            logger.info('Splitting dirty data has been successful. '
                        'Original data duplicates: '
                        f'{str(sum(df_dirty.duplicated()))}')

        helps.save_dfs(save_dict, data)
    else:
        logger.info('User aborted splitting.')


def compute_complete_dep_detector(data, save=False, dry_run=False):
    """
    Find dependencies in a table using Auto-ML.

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = opt.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='complete', dry_run=dry_run)
    end = timeit.default_timer()
    t = end - start
    result = {'time': t,
              'dep_optimizer': dep_optimizer}
    if save:
        print('Time: '+str(t))
        path = data.results_path + "dep_detector_complete_object.p"
        helps.save_pickle(result, path)
    else:
        print('\n~~~~~~~~~~~~~~~~~~~~\n')
        print(result['dep_optimizer'].print_trees())


def compute_greedy_dep_detector(data, save=False, dry_run=False):
    """ Find dependencies on a relational database table using a ML
    classifier (Datawig).

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = opt.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='greedy', dry_run=dry_run)
    end = timeit.default_timer()
    t = end - start
    print('Time: '+str(t))
    result = {'time': t,
              'dep_optimizer': dep_optimizer}
    if save:
        path = data.results_path + "dep_detector_greedy_object.p"
        helps.save_pickle(result, path)
    else:
        return result


def main(args):
    logger = logging.getLogger('pfd')
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    # create file handler with debug log level
    fh = logging.FileHandler('z_computer.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create console handler with a info log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    def no_valid_model(*args):
        logger.error("No valid model selected.")
        print("Select one of the following models with the --model flag:")
        for key in list(models.keys()):
            print(key)

    def no_valid_data():
        logger.error("No valid dataset selected.")
        print("Select one of the following datasets with the --data flag:")
        for key in list(datasets.keys()):
            print(key)

    datasets = {
        c.IR.title: c.IR,
        c.ADULT.title: c.ADULT,
        c.ABALONE.title: c.ABALONE,
        c.BALANCESCALE.title: c.BALANCESCALE,
        c.BREASTCANCER.title: c.BREASTCANCER,
        c.BRIDGES.title: c.BRIDGES,
        c.CERVICAL_CANCER.title: c.CERVICAL_CANCER,
        c.ECHOCARDIOGRAM.title: c.ECHOCARDIOGRAM,
        c.HEPATITIS.title: c.HEPATITIS,
        c.HORSE.title: c.HORSE,
        c.IRIS.title: c.IRIS,
        c.LETTER.title: c.LETTER,
        c.NURSERY.title: c.NURSERY,
        c.FLIGHTS.title: c.FLIGHTS,
        c.HOSPITAL_1k.title: c.HOSPITAL_1k,
        c.HOSPITAL_10k.title: c.HOSPITAL_10k,
        c.HOSPITAL_100k.title: c.HOSPITAL_100k
    }

    models = {'split': split_dataset,
              'complete_detect': compute_complete_dep_detector,
              'greedy_detect': compute_greedy_dep_detector,
              'manual_pfd': manual_pfd,
              'binary_pfd': binary_pfd,
              'linear_pfd': linear_pfd,
              'jump_pfd': jump_pfd,
              'clean_data': clean_data
              }
    detect_models = ['greedy_detect', 'complete_detect']

    dataset = datasets.get(args.dataset, no_valid_data)
    if callable(dataset):  # no valid dataname
        dataset()
    else:
        calc_fun = models.get(args.model, no_valid_model)
        if args.model in detect_models:
            logger.info(f'Running model {args.model} on data {args.dataset}'
                        'in a dry run.')
            calc_fun(dataset, args.save_result, args.dry_run)
        else:
            logger.info(f'Running model {args.model} on data {args.dataset}')
            calc_fun(dataset, args.save_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-c', '--column')
    parser.add_argument('-svr', '--save_result', action='store_true')
    parser.add_argument('-dry', '--dry_run', action='store_true')

    args = parser.parse_args()
    main(args)
