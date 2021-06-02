import os
import timeit
import argparse
import pickle
import pandas as pd
import lib.fd_imputer as fd
import lib.dep_detector as dep
import lib.constants as c


def save_pickle(obj, path):
    """ Pickles object obj and saves it to path. If path doesn't exist,
    creates path. """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    pickle.dump(obj, open(path, "wb"))
    message = '{0} successfully saved to {1}!'.format(
        os.path.basename(path).split('.')[0],
        path)
    print(message)


def split_dataset(data, save=True):
    """Splits a dataset into train, validate and test subsets.
    Be cautious when using, this might overwrite existing data"""
    print('You are about to split dataset ' + data.title)
    print('This might overwrite and nullify existing results.')
    sure = input('Do you want to proceed? [y/N]')
    if sure == 'y':
        df = pd.read_csv(data.data_path, sep=data.original_separator,
                         header=None)
        fd.split_df(data.title, df, [0.8, 0.1, 0.1], data.splits_path)
        print('successfully split.')
        print('original data duplicates: ' + str(sum(df.duplicated())))
    else:
        print('Aborted')


def compute_complete_dep_detector(data, save=False, set_dry_run=False):
    """
    Find dependencies in a table using Auto-ML.

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = dep.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='complete', dry_run=set_dry_run)
    end = timeit.default_timer()
    t = end - start
    result = {'time': t,
              'dep_optimizer': dep_optimizer}
    if save:
        print('Time: '+str(t))
        path = data.results_path + "dep_detector_complete_object.p"
        save_pickle(result, path)
    else:
        print('\n~~~~~~~~~~~~~~~~~~~~\n')
        print(result['dep_optimizer'].print_trees())


def compute_greedy_dep_detector(data, save=False, set_dry_run=False):
    """ Find dependencies on a relational database table using a ML
    classifier (Datawig).

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = dep.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='greedy', dry_run=set_dry_run)
    end = timeit.default_timer()
    t = end - start
    print('Time: '+str(t))
    result = {'time': t,
              'dep_optimizer': dep_optimizer}
    if save:
        path = data.results_path + "dep_detector_greedy_object.p"
        save_pickle(result, path)
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
        c.NURSERY.title: c.NURSERY,
        c.ABALONE.title: c.ABALONE,
        c.BALANCESCALE.title: c.BALANCESCALE,
        c.BREASTCANCER.title: c.BREASTCANCER,
        # c.BRIDGES.title: c.BRIDGES, mixed dtypes in col 3
        c.CHESS.title: c.CHESS,
        # c.ECHOCARDIOGRAM.title: c.ECHOCARDIOGRAM, mixed dtypes in col 1
        # c.HEPATITIS.title: c.HEPATITIS, mixed dtypes in col 17
        # c.HORSE.title: c.HORSE, mixed dtypes in unknown col
        c.IRIS.title: c.IRIS,
        c.LETTER.title: c.LETTER,
        c.MOVIES_DURATION.title: c.MOVIES_DURATION,
        c.MOVIES_ORDERING.title: c.MOVIES_ORDERING
    }

    models = {'split': split_dataset,
              'complete_detect': compute_complete_dep_detector,
              'greedy_detect': compute_greedy_dep_detector}
    detect_models = ['greedy_detect', 'complete_detect']

    dataset = datasets.get(args.dataset, no_valid_data)
    if callable(dataset):  # no valid dataname
        dataset()
    else:
        calc_fun = models.get(args.model, no_valid_model)
        if args.model in detect_models:
            calc_fun(dataset, args.save_result, args.set_dry_run)
        else:
            calc_fun(dataset, args.save_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-c', '--column')
    parser.add_argument('-svr', '--save_result', action='store_true')
    parser.add_argument('-dry', '--set_dry_run', action='store_true')

    args = parser.parse_args()
    main(args)
