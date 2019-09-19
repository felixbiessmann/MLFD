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


def print_fd_imputer_stats(datasets):
    """Prints fd imputer stats for all datasets."""
    import numpy as np
    print('\n\n~~~FD Imputer Statistical Analysis~~~\n')
    for data in datasets:
        path_to_res = data.results_path+'fd_imputer_results.p'
        fd_imputer_res = pickle.load(open(path_to_res, 'rb'))
        f1_scores = [y['f1'] for rhs in fd_imputer_res
                     for y in fd_imputer_res[rhs]
                     if ('f1' in y.keys())]
        mse_res = [y['mse'] for rhs in fd_imputer_res
                      for y in fd_imputer_res[rhs]
                      if ('mse' in y.keys())]
        mse_scores = [mse for mse in mse_res if (mse != '')]
        mse_no_val = [mse for mse in mse_res if (mse == '')]
        if len(mse_scores) > 0:
            mse_mean = np.mean(mse_scores)
            mse_min = min(mse_scores)
            mse_max = max(mse_scores)
        else:
            mse_mean = '-'
            mse_min = '-'
            mse_max = '-'
        fds = fd.read_fds(data.fd_path)
        no_fds = len([lhs for rhs in fds for lhs in fds[rhs]])
        print(data.title.upper())
        print('#FDs on Train-Split: {}'.format(no_fds))
        print('Mean f1-score: {}'.format(np.mean(f1_scores)))
        print('Minimal f1-score: {}'.format(min(f1_scores)))
        print('Maximal f1-score: {}'.format(max(f1_scores)))
        print('Mean mse: {}'.format(mse_mean))
        print('Minimal mse: {}'.format(mse_min))
        print('Maximal mse: {}'.format(mse_max))
        print('No result mse: {}'.format(len(mse_no_val)))
        print('~~~~~~~~~~')


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


def generate_random_fds(data, n=10, save=True):
    """ Generates n random FDs"""
    print('You are about to create random FDs for dataset ' + data.title)
    print('This might overwrite and nullify existing results.')
    sure = input('Do you want to proceed? [y/N]')
    if sure == 'y':
        df_train, df_validate, df_test = fd.load_dataframes(
            data.splits_path,
            data.title,
            data.missing_value_token)
        rand_fds = fd.random_dependency_generator(
            list(df_test.columns), n)
        pickle.dump(rand_fds, open(data.random_fd_path, "wb"))
        print('random FDs successfully written.')


def compute_dep_detector_lhs_stability(data, column, save=False):
    """ Trains SimpleImputer with a range from 3-15 training-cycles and
    stores the results of """
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    fd.check_split_for_duplicates([df_train, df_validate, df_test])

    Optimizer = dep.DepOptimizer(data)
    Optimizer.load_data()
    Optimizer.init_roots()
    col = Optimizer.roots[int(column)]
    minimal_lhs = {}
    for no_cycles in range(3, 16):
        print('training for {} cycles'.format(no_cycles))
        col.known_scores = {}
        col.cycles = no_cycles
        col.run_top_down(strategy='complete', dry_run=False)
        minimal_lhs[no_cycles] = col.extract_minimal_deps()

    if save:
        p = data.results_path + "dep_detector_lhs_stability.p"
        save_pickle(minimal_lhs, p)
    else:
        return minimal_lhs


def compute_rand_overfit_ml_imputer(data, no_dependencies=10, save=False):
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    df_overfit_train = pd.concat([df_train, df_validate])
    fd.check_split_for_duplicates([df_overfit_train, df_test])

    random_dep = pickle.load(open(data.random_fd_path, 'rb'))

    results = fd.run_ml_imputer_on_fd_set(df_overfit_train,
                                          df_overfit_train,
                                          df_test,
                                          random_dep,
                                          data.continuous)

    if save:
        p = data.results_path + "random_overfit_ml_imputer_results.p"
        save_pickle(results, p)
    else:
        return results


def compute_random_ml_imputer(data, no_dependencies=10, save=False):
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    fd.check_split_for_duplicates([df_train, df_validate, df_test])
    random_dep = pickle.load(open(data.random_fd_path, 'rb'))

    results = fd.run_ml_imputer_on_fd_set(df_train,
                                          df_validate,
                                          df_test,
                                          random_dep,
                                          data.continuous)

    if save:
        path = data.results_path + "random_ml_imputer_results.p"
        save_pickle(results, path)
    else:
        return results


def compute_overfit_ml_imputer(data, save=False):
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    df_overfit_train = pd.concat([df_train, df_validate])
    fd.check_split_for_duplicates([df_overfit_train, df_test])
    fds = fd.read_fds(data.fd_path)

    # overfitting is train, test, validate on the same set of data
    overfit_ml_imputer_results = fd.run_ml_imputer_on_fd_set(
        df_overfit_train,
        df_overfit_train,
        df_overfit_train,
        fds,
        data.continuous)
    if save:
        path = data.results_path + "overfitted_ml_results.p"
        save_pickle(overfit_ml_imputer_results, path)
    else:
        return overfit_ml_imputer_results


def compute_ml_imputer(data, save=False):
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    fd.check_split_for_duplicates([df_train, df_validate, df_test])
    fds = fd.read_fds(data.fd_path)

    ml_imputer_results = fd.run_ml_imputer_on_fd_set(df_train,
                                                     df_validate,
                                                     df_test,
                                                     fds,
                                                     data.continuous)
    if save:
        path = data.results_path + "ml_imputer_results.p"
        save_pickle(ml_imputer_results, path)
    else:
        return ml_imputer_results


def compute_complete_dep_detector(data, save=False):
    """ Find dependencies on a relational database table using a ML
    classifier (Datawig).

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = dep.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='complete', dry_run=False)
    end = timeit.default_timer()
    t = end - start
    print('Time: '+str(t))
    result = {'time': t,
              'dep_optimizer': dep_optimizer}
    if save:
        path = data.results_path + "dep_detector_complete_object.p"
        save_pickle(result, path)
    else:
        return result


def compute_greedy_dep_detector(data, save=False):
    """ Find dependencies on a relational database table using a ML
    classifier (Datawig).

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the whole DepOptimizer object to
    data.results_path or return it.
    """
    start = timeit.default_timer()
    dep_optimizer = dep.DepOptimizer(data, f1_threshold=0.9)
    dep_optimizer.search_dependencies(strategy='greedy', dry_run=False)
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


def compute_fd_imputer(data, save=False):
    """Compute the performance of the fd_imputer on a dataset.

    Keyword Arguments:
    data -- a dataset object from constants.py to perform computation upon
    save -- boolean, either save the result-dictionary to data.results_path
    or return it.
    """
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    fd.check_split_for_duplicates([df_train, df_validate, df_test])
    fds = fd.read_fds(data.fd_path)
    print(fds)

    df_fd_train = pd.concat([df_train, df_validate])
    fd_imputer_results = fd.run_fd_imputer_on_fd_set(df_fd_train,
                                                     df_test,
                                                     fds,
                                                     data.continuous)
    if save:
        path = data.results_path + "fd_imputer_results.p"
        save_pickle(fd_imputer_results, path)
    else:
        return fd_imputer_results


def main(args):
    """ Calculate results for the Master Thesis. """

    # this appears to be neccessary to avoid 'too many open files'-errors
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32768, hard))

    def no_valid_model(data, save):
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
        c.LETTER.title: c.LETTER
    }

    models = {'fd_imputer': compute_fd_imputer,
              'ml_imputer': compute_ml_imputer,
              'overfit_ml_imputer': compute_overfit_ml_imputer,
              'random_ml_imputer': compute_random_ml_imputer,
              'random_overfit_ml_imputer': compute_rand_overfit_ml_imputer,
              'split': split_dataset,
              'rand_fds': generate_random_fds,
              'complete_detect': compute_complete_dep_detector,
              'greedy_detect': compute_greedy_dep_detector,
              'dep_lhs_stability': compute_dep_detector_lhs_stability,
              'fd_imputer_stats': print_fd_imputer_stats}

    if (args.cluster_mode):
        for dataset in datasets.values():
            compute_complete_dep_detector(dataset, save=True)
            compute_greedy_dep_detector(dataset, save=True)

    else:
        data = datasets.get(args.data, no_valid_data)
        if callable(data):  # no valid dataname
            data()
        else:
            calc_fun = models.get(args.model, no_valid_model)
            if (args.model != 'dep_lhs_stability') and (args.model != 'fd_imputer_stats'):  # ugly but works
                calc_fun(data, save=True)
            elif args.model == 'dep_lhs_stability':
                calc_fun(data, column=args.column, save=True)
            elif args.model == 'fd_imputer_stats':
                print_fd_imputer_stats(datasets.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--data')
    parser.add_argument('-c', '--column')
    parser.add_argument('-cl', '--cluster_mode')

    args = parser.parse_args()
    main(args)
