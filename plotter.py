import os
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
import lib.fd_imputer as fd
import lib.plot_utils as pu
import lib.constants as c


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

    Pass -s img_name.pdf to export the graph to data.figures_path.
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

    datasets = {
        c.ADULT.title: c.ADULT,
        c.NURSERY.title: c.NURSERY,
        c.ABALONE.title: c.ABALONE,
        c.BALANCESCALE.title: c.BALANCESCALE,
        c.BREASTCANCER.title: c.BREASTCANCER,
        c.CHESS.title: c.CHESS,
        c.IRIS.title: c.IRIS,
        c.LETTER.title: c.LETTER
    }
    plots = {'f1_fd_imputer': plot_f1_fd_imputer,
             'f1_ml_imputer': plot_f1_ml_imputer,
             'f1_ml_fd': plot_f1_ml_fd,
             'mse_ml_fd': plot_mse_ml_fd,
             'f1_ml_overfit': plot_f1_ml_overfit,
             'f1_random_overfit': plot_f1_random_ml_overfit,
             'dep_detector_lhs_stability': plot_dep_detector_lhs_stability}

    data = datasets.get(args.data, no_valid_data)
    if args.all:
        for data in datasets.values():
            for plot_name in plots:
                plot_fun = plots[plot_name]
                try:
                    fig, ax = plot_fun(data)
                    ax.set_axisbelow(True)
                    plt.tight_layout()
                    path = data.figures_path + plot_name + '.pdf'
                    pu.save_fig(fig, path)
                    plt.close(fig)
                except OSError:
                    '''Could not create {0}-plot for dataset
                    {1}: File not found.'''.format(plot_name, data.title)
    else:
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


def plot_dep_detector_lhs_stability(data):
    """ Plots the dependency of lhs-stability on the amount of training-cycles
    Datawig is trained with for the fifth column of the iris dataset as RHS."""
    minimal_lhs_dict = load_result(
        data.results_path+'dep_detector_lhs_stability.p')
    cycle_lhs = {}
    last_cycle = 0  # search highest no of cycles
    for cycle in minimal_lhs_dict:
        cycle_lhs[cycle] = [lhs for lhs in minimal_lhs_dict[cycle]]
        if cycle > last_cycle:
            last_cycle = cycle
            final_lhs = cycle_lhs[cycle]

    distances = {}
    for cycle in cycle_lhs:
        distance = 0
        for lhs in cycle_lhs[cycle]:
            if lhs not in final_lhs:
                distance += 1
        distances[cycle] = int(distance)

    cycles, dist = list(distances.keys()), list(distances.values())
    print(dist)
    print(cycles)
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)
    ax.plot(cycles, dist)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set(title='Stability-Analysis on '+data.title.capitalize(),
           xlabel='Training-Cycles',
           ylabel='Undetected minimal LHSs')
    return(fig, ax)


def plot_f1_fd_imputer(data):
    fd_imputer_res = load_result(data.results_path+'fd_imputer_results.p')
    res_bigger_zero = [(y['f1'],
                        sorted(list(map(int, y['lhs']))),
                        str(rhs))
                       for rhs in fd_imputer_res
                       for y in fd_imputer_res[rhs]
                       if ('f1' in y.keys())]

    res_bigger_zero = [(res[0],
         ''.join(str(res[1])[1:-1]).replace('\'', '')+r'$\rightarrow$'+str(res[2]))
         for res in res_bigger_zero]

    res_bigger_zero.sort()
    print(res_bigger_zero)

    f1_fd, lhs_names = zip(*res_bigger_zero[-10:])

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 6)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)
    ax.barh(lhs_names, f1_fd)

    ax.set(title='FD Imputer Performance on '+data.title.capitalize(),
           xlabel='F1-Score',
           xlim=[0.0, 1.0])
    return(fig, ax)


def plot_f1_ml_imputer(data):
    ml_imputer_res = load_result(data.results_path+'ml_imputer_results.p')

    res_bigger_zero = [(y['f1'],
                        sorted(list(map(int, y['lhs']))),
                        str(rhs))
                       for rhs in ml_imputer_res
                       for y in ml_imputer_res[rhs]
                       if ('f1' in y.keys())]

    res_bigger_zero = [(res[0],
         ''.join(str(res[1])[1:-1]).replace('\'', '')+r'$\rightarrow$'+str(res[2]))
         for res in res_bigger_zero]

    res_bigger_zero.sort()
    print(res_bigger_zero)

    f1_fd, lhs_names = zip(*res_bigger_zero[-10:])

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)
    ax.barh(lhs_names, f1_fd)

    ax.set(title='ML Imputer Performance on '+data.title.capitalize(),
           xlabel='F1-Score',
           xlim=[0.0, 1.0])
    return(fig, ax)


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
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.scatter(f1_overfit, f1_ml, c='C0')
    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), c='C1', linewidth=1)
    ax.set(title='Classification Performance on ' + data.title.capitalize(),
           xlabel='F1-Score ML Imputer Overfitted',
           ylabel='F1-Score ML Imputer',
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
    fig_size = pu.get_fig_size(10, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    ax.scatter(f1_overfit, f1_ml, c='C0')
    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), c='C1')
    ax.set(title='Classification Performance on Random FDs '+data.title.capitalize(),
           xlabel='F1-Score ML Imputer Overfitted',
           ylabel='F1-Score ML Imputer',
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

    ax.scatter(list(range(0, len(rel_mse))), rel_mse, c='C0')
    ax.plot(np.linspace(-2, len(rel_mse)), [1]*50, c='C1')

    ax.set(title='Mean Squared Error of Two Imputers on '+data.title.capitalize(),
           xlabel='FD LHS Combination',
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

    ax.scatter(f1_fd, f1_ml, color='C0')
    ax.plot(np.linspace(-2, 2), np.linspace(-2, 2), lw=pu.plot_lw(),
            color='C1')

    ax.set(title='Classification Performance on '+data.title.capitalize(),
           xlabel='F1-Score FD Imputer',
           ylabel='F1-Score ML Imputer',
           xlim=[-0.1, 1.1],
           ylim=[-0.1, 1.1])
    return(fig, ax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=main.__doc__)
    # 'Script for plotting figures for Philipp Jung\'s Master Thesis.\n')

    parser.add_argument(
        '-s', '--save', help='specify filename and -type to save the figure.')
    parser.add_argument('-f', '--figure', help='specify a figure to plot.')
    parser.add_argument(
        '-d', '--data', help='specify a dataset to use results of.')
    parser.add_argument('-a', '--all', help='plot and save all possible plots',
            action='store_true')

    args = parser.parse_args()
    main(args)
