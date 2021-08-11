import timeit
import logging
import argparse
import pandas as pd
import lib.helpers as helps
import lib.optimizer as opt
import lib.constants as c


def automatic_pfd(data, save=False, dry_run=False):
    """
    Uses feature permutation to automatically search for minimal PFDs.
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
    measured_performance = 0
    threshold = 1

    # TODO this part doesn't work yet -- perfect predictor has accuracy 1
    while measured_performance < threshold:
        exclude_cols = [c for c in df_train.columns if c not in include_cols]
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

        def map_index(x):
            """Makes column list index human-readable"""
            return f'{x} ({data.column_map[x]})'

        df_importance.index = df_importance.index.map(map_index)
        df_imp = df_importance.iloc[:, :1]
        print(df_imp)
        print(f"What's your threshold for {metric}?")
        user_threshold = float(input(''))

        # the margin is how much performance we can shave off
        margin = measured_performance - user_threshold
        df_importance_cumsum = df_imp.sort_values(
                'importance', ascending=True).cumsum()
        se_importance_distance = df_importance_cumsum.loc[:, 'importance'] - margin

        exclude_cols = [x[0][0] for x in se_importance_distance.iteritems()
                        if x[1] < 0]

        include_cols = [c for c in include_cols if c not in exclude_cols]


def manual_pfd(data, save=False, dry_run=False):
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
    logger.debug(f'User chose rhs_index {rhs_index}')
    include_cols = list(df_train.columns)
    while True:
        exclude_cols = [c for c in df_train.columns if c not in include_cols]
        logger.info(f"Begin predictor training with LHS {include_cols}")
        df_importance = opt.iterate_pfd(include_cols,
                                        df_train,
                                        df_validate,
                                        df_test,
                                        rhs_index)

        print("Found the following importances via feature permutation:")

        def map_index(x):
            """Makes column list index human-readable"""
            return f'{x} ({data.column_map[x]})'

        df_importance.index = df_importance.index.map(map_index)
        print(df_importance.iloc[:, :1])
        # print("What's your threshold for {metric}?")
        print(f'Excluded: {exclude_cols}')
        print("Which columns do you want to exclude? (q to quit)")
        i = input('')
        if i == 'q':
            break
        exclude_cols = [int(c) for c in i.split(',')]

        include_cols = [c for c in include_cols if c not in exclude_cols]


def split_dataset(data, save=True):
    """Splits a dataset into train, validate and test subsets.
    Be cautious when using, this might overwrite existing data"""
    print(f'''You are about to split dataset {data.title}.
If you continue, splits will be saved to {data.splits_path}.
This might overwrite and nullify existing results.
Do you want to proceed? [y/N]''')
    sure = input('')
    if sure == 'y':
        df = pd.read_csv(data.data_path, sep=data.original_separator,
                         header=None)
        splits_path = ''
        if save:
            splits_path = data.splits_path
        helps.split_df(data.title, df, (0.8, 0.1, 0.1), splits_path)
        print('Splitting successful.')
        logger = logging.getLogger('pfd')
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
              'automatic_pfd': automatic_pfd,
              'manual_pfd': manual_pfd}
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
