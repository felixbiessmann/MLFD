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


def print_ml_imputer_stats(datasets):
    """ Prints ml imputer stats for all datasets."""
    import numpy as np
    print('\n\n~~~ML Imputer Statistical Analysis~~~\n')
    datasets = list(datasets)
    datasets.remove(c.BREASTCANCER)  # keine results
    for data in datasets:
        train, validate, test = fd.load_dataframes(data.splits_path,
                data.title,
                data.missing_value_token)
        no_test_rows = test.shape[0]
        path_to_res = data.results_path+'ml_imputer_results.p'
        ml_imputer_res = pickle.load(open(path_to_res, 'rb'))
        ml_no_imputations = 0
        sequential_fds = 0
        ml_f1_scores = []
        ml_mse_scores = {}
        for rhs in ml_imputer_res:
            for y in ml_imputer_res[rhs]:
                if 'f1' in y.keys():
                    ml_f1_scores.append(y['f1'])
                elif 'mse' in y.keys():
                    sequential_fds += 1
                    if y['mse'] == '':
                        ml_no_imputations += 1
                    else:
                        ml_mse_scores.setdefault(rhs, []).append(y['mse'])


        ml_mse_stats = {}
        if len(ml_mse_scores) > 0:
            for rhs in ml_mse_scores:
                ml_mse_stats[rhs] = {'variance': np.var(ml_mse_scores[rhs]),
                                     'mean': np.mean(ml_mse_scores[rhs]),
                                     'min': min(ml_mse_scores[rhs]),
                                     'max': max(ml_mse_scores[rhs])}

        fds = fd.read_fds(data.fd_path)
        no_fds = len([lhs for rhs in fds for lhs in fds[rhs]])
        print(data.title.upper())
        print('#FDs on Train-Split: {}'.format(no_fds))
        print('Mean F1-Score: {}'.format(np.mean(ml_f1_scores)))
        print('Maximal F1-Score: {}'.format(max(ml_f1_scores)))
        print('Minimal F1-Score: {}'.format(min(ml_f1_scores)))
        print('#FDs where f1=0: {}'.format(len(
            [f1 for f1 in ml_f1_scores if f1 == 0])))
        print('\nMSE ANALYSIS')
        print('# Sequential FDs: {}'.format(sequential_fds))
        print('Number of rows in test subset: {}'.format(no_test_rows))
        print('Sequential FD with no imputation: {}'.format(ml_no_imputations))
        for rhs in ml_mse_stats:
            print('  MSE RHS {}'.format(rhs))
            print('  Var MSE: {}'.format(ml_mse_stats[rhs]['variance']))
            print('  Mean MSE: {}'.format(ml_mse_stats[rhs]['mean']))
            print('  Min MSE: {}'.format(ml_mse_stats[rhs]['min']))
            print('  Max MSE: {}\n'.format(ml_mse_stats[rhs]['max']))
        print('~~~~~~~~~~')


def print_fd_imputer_stats(datasets):
    """ Prints fd imputer stats for all datasets."""
    import numpy as np
    print('\n\n~~~FD Imputer Statistical Analysis~~~\n')
    datasets = list(datasets)
    datasets.remove(c.BREASTCANCER)  # keine results
    for data in datasets:
        train, validate, test = fd.load_dataframes(data.splits_path,
                data.title,
                data.missing_value_token)
        no_test_rows = test.shape[0]
        path_to_res = data.results_path+'fd_imputer_results.p'
        fd_imputer_res = pickle.load(open(path_to_res, 'rb'))
        fd_no_imputations = 0
        sequential_fds = 0
        fd_f1_scores = []
        fd_mse_scores = {}
        fd_mse_nans = []
        for rhs in fd_imputer_res:
            for y in fd_imputer_res[rhs]:
                if 'f1' in y.keys():
                    fd_f1_scores.append(y['f1'])
                elif 'mse' in y.keys():
                    sequential_fds += 1
                    fd_mse_nans.append(y['nans'])
                    if y['mse'] == '':
                        fd_no_imputations += 1
                    else:
                        fd_mse_scores.setdefault(rhs, []).append(y['mse'])

        fd_mse_stats = {}
        if len(fd_mse_scores) > 0:
            fd_mse_mean_nans = sum(fd_mse_nans)/len(fd_mse_nans)
            mean_imp_coverage = 100*(1 - fd_mse_mean_nans / no_test_rows)
            for rhs in fd_mse_scores:
                fd_mse_stats[rhs] = {'variance': np.var(fd_mse_scores[rhs]),
                                     'mean': np.mean(fd_mse_scores[rhs]),
                                     'min': min(fd_mse_scores[rhs]),
                                     'max': max(fd_mse_scores[rhs])}
        else:
            fd_mse_mean_nans = 'No missing MSE imputations.'
            mean_imp_coverage = '-'

        fds = fd.read_fds(data.fd_path)
        no_fds = len([lhs for rhs in fds for lhs in fds[rhs]])
        print(data.title.upper())
        print('#FDs on Train-Split: {}'.format(no_fds))
        print('Mean F1-Score: {}'.format(np.mean(fd_f1_scores)))
        print('Maximal F1-Score: {}'.format(max(fd_f1_scores)))
        print('Minimal F1-Score: {}'.format(min(fd_f1_scores)))
        print('#FDs where f1=0: {}'.format(len(
            [f1 for f1 in fd_f1_scores if f1 == 0])))
        print('\nMSE ANALYSIS')
        print('#Sequential FDs: {}'.format(sequential_fds))
        print('Sequential FD with no imputation: {}'.format(fd_no_imputations))
        print('Mean missing MSE-imputations per FD: {}'.format(fd_mse_mean_nans))
        print('Number of Rows in Test Subset: {}'.format(no_test_rows))
        print('Mean Imputation Coverage: {}%'.format(mean_imp_coverage))
        for rhs in fd_mse_stats:
            print('  MSE RHS {}'.format(rhs))
            print('  Var MSE: {}'.format(fd_mse_stats[rhs]['variance']))
            print('  Mean MSE: {}'.format(fd_mse_stats[rhs]['mean']))
            print('  Min MSE: {}'.format(fd_mse_stats[rhs]['min']))
            print('  Max MSE: {}\n'.format(fd_mse_stats[rhs]['max']))
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
    stores the results."""
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
    df_overfit_train = pd.concat([df_train, df_validate, df_test])
    fd.check_split_for_duplicates([df_overfit_train])
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
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, hard))

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
              'fd_imputer_stats': print_fd_imputer_stats,
              'ml_imputer_stats': print_ml_imputer_stats}
    special_models = ['dep_lhs_stability',
                      'fd_imputer_stats',
                      'ml_imputer_stats']

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
            if args.model not in special_models:
                calc_fun(data, save=True)
            elif args.model == 'dep_lhs_stability':
                calc_fun(data, column=args.column, save=True)
            elif args.model == 'fd_imputer_stats':
                print_fd_imputer_stats(datasets.values())
            elif args.model == 'ml_imputer_stats':
                print_ml_imputer_stats(datasets.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--data')
    parser.add_argument('-c', '--column')
    parser.add_argument('-cl', '--cluster_mode')

    args = parser.parse_args()
    main(args)
