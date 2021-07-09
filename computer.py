import timeit
import argparse
import pandas as pd
import lib.helpers as helps
import lib.optimizer as opt
import lib.constants as c


def compute_pfd(data, save=False, dry_run=False):
    """
    Use SHAP to compute the PFD of a dataset. The user is prompted to
    insert how much mean absolute deviation from the mean function value
    they want to secrifice for a smaller LHS
    """
    print("Computing PFD. What is the index of the RHS to investigate?")
    label = int(input(''))  # make sure to select cols based on integers

    df_train, df_validate, df_test = helps.load_splits(data.splits_path,
                                                       data.title,
                                                       ',')

    include_cols = list(df_train.columns)
    while True:
        exclude_cols = [c for c in df_train.columns if c not in include_cols]
        print("Training model...")
        df_importance = opt.iterate_pfd(include_cols,
                                        df_train,
                                        df_validate,
                                        df_test,
                                        label)

        print("Found the following importances via feature permutation:")
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
        print('successfully split.')
        print('original data duplicates: ' + str(sum(df.duplicated())))
    else:
        print('Aborted')


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
    # this appears to be neccessary to avoid 'too many open files'-errors
    # import resource

    # The two following lines set the max number of open files. However,
    # this does not work on Mac OSX, which is why they're commented out
    # for now.

    # soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (100000, hard))

    def no_valid_model(*args):
        print("No valid model. Please specify one of the following models:")
        for key in list(models.keys()):
            print(key)

    def no_valid_data():
        print("No valid dataset selected. Specify one of the following:")
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
              'compute_pfd': compute_pfd}
    detect_models = ['greedy_detect', 'complete_detect']

    dataset = datasets.get(args.dataset, no_valid_data)
    if callable(dataset):  # no valid dataname
        dataset()
    else:
        calc_fun = models.get(args.model, no_valid_model)
        if args.model in detect_models:
            calc_fun(dataset, args.save_result, args.dry_run)
        else:
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
