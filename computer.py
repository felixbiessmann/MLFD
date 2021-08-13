import numpy as np
import timeit
import logging
import argparse
import pandas as pd
import lib.helpers as helps
import lib.optimizer as opt
import lib.constants as c


def manual_pfd(data, *args, **kwargs):
    """
    Use feature permutation to compute the PFD of a dataset. The user is
    prompted to insert how much mean absolute deviation from the mean
    function value they want to secrifice for a smaller LHS.
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start manual search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')
    print("Compute a PFD.")
    print("What is the index of the RHS to investigate?")
    print(repr(data.column_map))
    rhs_index = int(input(''))  # select columns based on integers
    logger.debug(f'User chose rhs_index {rhs_index}.')

    exclude_cols = []
    include_cols = list(df_train.columns)
    lhs = [c for c in include_cols if c != rhs_index]
    logger.info("Begin global predictor training with complete LHS.")
    df_imp, measured_performance, metric = opt.get_importance_pfd(df_train,
                                                                  df_validate,
                                                                  df_test,
                                                                  rhs_index)
    df_imp['description'] = df_imp.index.to_series().apply(lambda x: data.column_map.get(x, 'NA'))

    while True:
        logger.info(f"Trained a predictor with {metric} "
                    f"{measured_performance}.")
        print("Found the following importances via feature permutation:")
        print(df_imp.loc[lhs, ['description', 'importance']].sort_values(
            'importance', ascending=False))
        print(f'Excluded: {exclude_cols}')

        print("Which columns do you want to exclude? (q to quit)")
        i = input('')
        if i == 'q':
            break

        exclude_cols = [int(c) for c in i.split(',')]
        include_cols = [c for c in include_cols if c not in exclude_cols]
        lhs = [c for c in include_cols if c != rhs_index]

        logger.info(f"Begin predictor training with LHS {lhs}")
        measured_performance = opt.iterate_pfd(include_cols,
                                               df_train,
                                               df_validate,
                                               df_test,
                                               rhs_index)


def jump_pfd(data, *args, **kwargs):
    """
    TODO Build like manual_pfd
    Uses feature permutation to automatically search for minimal PFDs.

    Currently, this approach doesn't work, because AG's feature_permutation
    returns feature importances that aren't additive and thus don't sum up to
    equal the model's loss -- I am only aware of shap.TreeExplainer's ability
    to explain the log_loss of a function. So this seems to be a dead end.

    In AutoGluon, higher performance metrics are always better. This leads to
    the circumstance that the MSE is negative!
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start automatical search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')
    print(repr(data.column_map))
    print("What is the index of the RHS to investigate?")
    rhs_index = int(input(''))  # select columns based on integers
    logger.debug(f'User chose rhs_index {rhs_index}')

    include_cols = list(df_train.columns)
    measured_performance = 1
    threshold = 0
    first_run = True

    while True:
        exclude_cols = [c for c in list(df_train.columns)
                        if c not in include_cols]
        logger.info("Begin predictor training")
        df_importance, performance, metric = opt.iterate_pfd(include_cols,
                                                             df_train,
                                                             df_validate,
                                                             df_test,
                                                             rhs_index)
        measured_performance = performance[metric]
        logger.info(
            f"Trained a predictor with {metric} {measured_performance}")
        print("Found the following importances via feature permutation:")
        if measured_performance < threshold:
            logger.info("The newly trained model's performance of "
                        f"{measured_performance} is below the thresold of "
                        f" {threshold}. Stopping the search.")
            break

        def map_index(x):
            """Makes column list index human-readable"""
            return f'{x} ({data.column_map[x]})'

        df_importance.index = df_importance.index.map(map_index)
        df_imp = df_importance.iloc[:, :1]
        print(df_imp)
        if first_run:
            print(f"What's your threshold for {metric}?")
            threshold = float(input(''))
            first_run = False

        # the margin is how much performance we can shave off
        margin = measured_performance - threshold
        if margin < 0:
            logger.info(f"The set threshold of {threshold} is below "
                        "the measured_performance of {measured_performance}."
                        "Stopping the search.")
            break

        df_importance_cumsum = df_imp.sort_values(
            'importance', ascending=True).cumsum()
        importance_distance = df_importance_cumsum.loc[:,
                                                       'importance'] - margin
        exclude_cols = [int(x[0][0]) for x in importance_distance.iteritems()
                        if x[1] < 0]
        include_cols = [c for c in include_cols if c not in exclude_cols]


def linear_pfd(data, *args, **kwargs):
    """
    TODO Build like manual_pfd
    Uses feature permutation to automatically search for minimal PFDs.

    In AutoGluon, higher performance metrics are always better. This leads to
    the circumstnace that the MSE is negative! So don't dispair!
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start automatical search of PFDs for dataset {data.title}")
    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')
    print(repr(data.column_map))
    print("What is the index of the RHS to investigate?")
    rhs_index = int(input(''))  # select columns based on integers
    logger.debug(f'User chose rhs_index {rhs_index}')

    include_cols = list(df_train.columns)
    measured_performance = 1
    threshold = 0
    first_run = True

    while True:
        logger.info("Begin predictor training")
        df_importance, performance, metric = opt.iterate_pfd(include_cols,
                                                             df_train,
                                                             df_validate,
                                                             df_test,
                                                             rhs_index)
        measured_performance = performance[metric]
        logger.info(
            f"Trained a predictor with {metric} {measured_performance}")
        if measured_performance < threshold:
            logger.info("The newly trained model's performance of "
                        f"{measured_performance} is below the threshold of "
                        f" {threshold}. Stopping the search.")
            break

        def map_index(x):
            """Makes column list index human-readable"""
            return f'{x} ({data.column_map[x]})'

        df_importance.index = df_importance.index.map(map_index)
        df_imp = df_importance.iloc[:, :1]
        print("Found the following importances via feature permutation:")
        print(df_imp)
        if first_run:
            print(f"What's your threshold for {metric}?")
            threshold = float(input(''))
            first_run = False

        # how much performance can we shave off?
        margin = measured_performance - threshold
        if margin < 0:
            logger.info(f"The set threshold of {threshold} is below "
                        "the measured_performance of {measured_performance}."
                        "Stopping the search.")
            break

        # the least important column
        exclude_col = int(df_imp.iloc[-1, :].name[0])
        include_cols = [c for c in include_cols if c != exclude_col]


def split_dataset(data, save=True):
    """Splits a dataset into train, validate and test subsets.
    Be cautious when using, this might overwrite existing data"""
    print(f'''You are about to split dataset {data.title}.
If you continue, splits will be saved to {data.splits_path}.
This might overwrite and nullify existing results.
Do you want to proceed? [y/N]''')
    logger = logging.getLogger('pfd')
    sure = input('')
    if sure == 'y':
        df = pd.read_csv(data.data_path, sep=data.original_separator,
                         header=None)
        splits_path = ''
        if save:
            splits_path = data.splits_path
        helps.split_df(data.title, df, (0.8, 0.1, 0.1), splits_path)
        print('Splitting successful.')
        logger.debug('Splitting has been successful.'
                     f'original data duplicates: {str(sum(df.duplicated()))}')
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
    }

    models = {'split': split_dataset,
              'complete_detect': compute_complete_dep_detector,
              'greedy_detect': compute_greedy_dep_detector,
              'manual_pfd': manual_pfd,
              'linear_pfd': linear_pfd,
              'jump_pfd': jump_pfd}
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
