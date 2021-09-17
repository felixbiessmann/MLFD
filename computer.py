import timeit
import random
import logging
import argparse
import pandas as pd
from typing import Tuple
import lib.helpers as helps
import lib.optimizer as opt
import lib.constants as c


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


def clean_data(data: c.Dataset, *args, **kwargs):
    """
    Train a model on the dirty (default) train-test split and calculate
    cleaning performance by comparing the dirty (default) validate-split
    to the clean validate-split.

    Depending on the type of comparison between the validate sets, this is
    either an error-detection experiment, or a cleaning experiment.
    """



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
    df_imp, measured_performance, metric, rhs = global_predictor_explained(
        data, df_train, df_validate, df_test)

    measured_performance = measured_performance
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
    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')

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
    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')

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

    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')

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
    import os
    print(f'''
          You are about to split dataset {data.title}.
          If you continue, splits will be saved to {data.splits_path}.
          This might overwrite and nullify existing results.
          Do you want to proceed? [y/N]''')
    logger = logging.getLogger('pfd')
    sure = input('')
    if sure == 'y':
        split_ratio = (0.8, 0.1, 0.1)
        splits_path = data.splits_path
        rndint = random.randint(0, 10000)
        df_clean = pd.read_csv(data.data_path, sep=data.original_separator,
                               header=None)
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
            df_dirty = pd.read_csv(data.dirty_data_path,
                                   sep=data.original_separator,
                                   header=None)
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
    else:
        logger.info('User aborted splitting.')


def cleaning_performance(data, save=False, dry_run=False):
    """
    Train a model on dirty data. Use it to calculate cleaning
    peformance by comparing y_pred to y_true from the clean data.
    """
    logger = logging.getLogger('pfd')
    logger.debug(f"Start automatical search of PFDs for dataset {data.title}")

    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')


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
