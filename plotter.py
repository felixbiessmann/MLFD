import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import lib.plot_utils as pu
from constants import ADULT, NURSERY
from constants import METANOME_DATA_PATH


def load_result(path_to_pickle):
    """ Returns a pickled result from RESULTS_PATH.
    Also checks when the result was last modified."""
    file_last_modified(path_to_pickle)
    return pickle.load(open(path_to_pickle, 'rb'))


def file_last_modified(path_to_file):
    """ Prints a string to indicate when a file has been
    modified the last time."""
    mod_time = time.localtime(os.path.getmtime(path_to_file))
    time_str = time.strftime("%a, %d. %b %Y %H:%M:%S", mod_time)
    print(time_str + ": Last change to " + os.path.basename(path_to_file))


def main(args):
    """ Plot figures for the Master Thesis. Just call this script
    and set an -f flag with appropriate value as used in the conditional
    statement below.

    Pass -s img_name.eps to export the graph to data.figures_path.
    """
    def no_valid_figure():
        print("No valid figure. Please select one from this list:")
        for key in list(plots.keys()):
            print(key)

    def no_valid_data():
        print("No valid dataset selected. Specify one of the following:")
        for key in list(datasets.keys()):
            print(key)
        return 0

    datasets = {'adult': ADULT,
                'nursery': NURSERY}
    plots = {'f1_ml_fd': plot_f1_ml_fd,
             'mse_ml_fd': plot_mse_ml_fd,
             'f1_ml_overfit': plot_f1_ml_overfit,
             'f1_random_overfit': plot_f1_random_ml_overfit}

    data = datasets.get(args.data, no_valid_data)
    if data != 0:
        plot_fun = plots.get(args.figure, no_valid_figure)

    if (plot_fun != no_valid_figure) and (data != 0):
        fig, ax = plot_fun(data)

        ax.set_axisbelow(True)
        plt.tight_layout()

        if args.save:
            pu.save_fig(fig, data.figures_path + args.save)
        else:
            plt.show()
    else:
        plot_fun()


def plot_f1_ml_overfit(data):
    ml_imputer_res = load_result(
            data.results_path+"ml_imputer_results.p")
    overf_ml_imputer_res = load_result(
            data.results_path+"overfitted_ml_results.p")

    f1_ml = [y['f1'] for x in ml_imputer_res for y in ml_imputer_res[x]
             if 'f1' in y.keys()]
    f1_overfit = [y['f1'] for x in overf_ml_imputer_res
                  for y in overf_ml_imputer_res[x]
                  if 'f1' in y.keys()]

    pu.figure_setup()
    fig_size = pu.get_fig_size(5, 10)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.scatter(f1_overfit, f1_ml)
    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), c='red')
    ax.set(title=data.title+'.csv Classification Performance',
           xlabel='f1-score ML imputer overfitted',
           ylabel='f1-score ML imputer',
           xlim=[-0.1, 1.1],
           ylim=[-0.1, 1.1])
    return (fig, ax)


def plot_f1_random_ml_overfit(data):
    ml_imputer_results = load_result(
        data.results_path+"random_ml_imputer_results.p")
    overf_ml_imputer_res = load_result(
        data.results_path+"random_overfit_ml_imputer_results.p")

    f1_ml = [y['f1'] for x in ml_imputer_results for y in ml_imputer_results[x]
             if 'f1' in y.keys()]
    f1_overfit = [y['f1'] for x in overf_ml_imputer_res
                  for y in overf_ml_imputer_res[x]
                  if 'f1' in y.keys()]

    pu.figure_setup()
    fig_size = pu.get_fig_size(5, 10)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.scatter(f1_overfit, f1_ml)
    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), c='red')
    ax.set(title=data.title+'.csv Classification Performance on Random FDs',
           xlabel='f1-score ML imputer overfitted',
           ylabel='f1-score ML imputer',
           xlim=[-0.1, 1.1],
           ylim=[-0.1, 1.1])
    return (fig, ax)


def plot_mse_ml_fd(data):
    fd_imputer_results = load_result(
            data.results_path+"fd_imputer_results.p")
    ml_imputer_results = load_result(
            data.results_path+"ml_imputer_results.p")

    mse_fd = [y['mse'] for x in fd_imputer_results
              for y in fd_imputer_results[x] if 'mse' in y.keys()]
    mse_ml = [y['mse'] for x in ml_imputer_results
              for y in ml_imputer_results[x] if 'mse' in y.keys()]

    rel_mse = []
    for i, x in enumerate(mse_fd):
        if x != '':
            rel_mse.append(mse_fd[i]/mse_ml[i])
        else:
            rel_mse.append(np.nan)

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.scatter(list(range(0, len(rel_mse))), rel_mse)
    ax.plot(np.linspace(-2, len(rel_mse)), [1]*50, c='red')

    ax.set(title=data.title+'.csv Mean Squared Error of two imputers',
           xlabel='FD LHS combination',
           ylabel='$MSE_{FD}$ / MSE ML',
           xlim=[-0.1, 11])
    return (fig, ax)


def plot_f1_ml_fd(data):
    fd_imputer_results = load_result(
            data.results_path+"fd_imputer_results.p")
    ml_imputer_results = load_result(
            data.results_path+"ml_imputer_results.p")

    f1_fd = [y['f1'] for x in fd_imputer_results
             for y in fd_imputer_results[x] if 'f1' in y.keys()]
    f1_ml = [y['f1'] for x in ml_imputer_results
             for y in ml_imputer_results[x] if 'f1' in y.keys()]

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), lw=pu.plot_lw(),
            color='red')
    ax.scatter(f1_fd, f1_ml)

    ax.set(title=data.title+'.csv Classification Performance',
           xlabel='f1-score FD imputer',
           ylabel='f1-score ML imputer',
           xlim=[-0.1, 1.1],
           ylim=[-0.1, 1.1])
    return(fig, ax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')
    parser.add_argument('-f', '--figure')
    parser.add_argument('-d', '--data')

    args = parser.parse_args()
    main(args)
