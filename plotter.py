import os
import time
import datawig
import argparse
from pathlib import Path
import lib.helpers as helps
import matplotlib.pyplot as plt
import numpy as np
import pickle
import lib.plot_utils as pu
import lib.constants as c
from sklearn.metrics import precision_recall_curve, auc


def main(args):
    """
    Plot figures for the Master Thesis. Just call this script
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
        c.IR.title: c.IR,
        c.ADULT.title: c.ADULT,
        c.NURSERY.title: c.NURSERY,
        c.ABALONE.title: c.ABALONE,
        c.BALANCESCALE.title: c.BALANCESCALE,
        c.BREASTCANCER.title: c.BREASTCANCER,
        c.IRIS.title: c.IRIS,
        c.LETTER.title: c.LETTER,
        c.HOSPITAL_1k.title: c.HOSPITAL_1k,
        c.HOSPITAL_10k.title: c.HOSPITAL_10k,
        c.HOSPITAL_100k.title: c.HOSPITAL_100k
    }

    plots = {
             'f1_cleaning_detection': plot_f1_cleaning_detection_local,
             'f1_cleaning_detection_global': plot_f1_cleaning_detection_global,
             'prec_rec_local': plot_prec_rec_local,
             'auc_cleaning_global': plot_auc_cleaning_global,
             'prec_threshold': plot_prec_threshold,
             }

    result_name = args.name
    data = datasets.get(args.data, no_valid_data)
    if args.all:  # plot everything and save it.
        raise NotImplementedError('Need to handle result_name.')
        # for data in datasets.values():
        #     for plot_name in plots:
        #         plot_fun = plots[plot_name]
        #         try:
        #             fig, ax = plot_fun(data, result_name)
        #             ax.set_axisbelow(True)
        #             plt.tight_layout()
        #             path = data.figures_path + plot_name + '.pdf'
        #             pu.save_fig(fig, path)
        #             plt.close(fig)
        #         except OSError:
        #             f'''Could not create {plot_name}-plot for dataset
        #             {data.title}: File not found.'''
    else:
        plot_fun = plots.get(args.figure, no_valid_figure)

        if ((plot_fun.__code__.co_code != no_valid_figure.__code__.co_code)
                and (data != 0)):  # plot one specific plot
            fig, ax = plot_fun(data, result_name)

            ax.set_axisbelow(True)
            plt.tight_layout()

            if args.save:
                pu.save_fig(fig, data.figures_path + args.save)
                print(f'Successfully saved the figure to {data.figures_path}.')
            else:
                plt.show()
        else:  # runs no_valid_figure()
            plot_fun()


def load_result(path_to_pickle: Path):
    """
    Returns a pickled result from RESULTS_PATH.
    Also checks when the result was last modified.
    """
    file_last_modified(path_to_pickle)
    return pickle.load(open(path_to_pickle, 'rb'))


def file_last_modified(path_to_file: Path):
    """
    Prints a string to indicate when a file has been
    modified the last time.
    """
    mod_time = time.localtime(os.path.getmtime(path_to_file))
    time_str = time.strftime("%a, %d. %b %Y %H:%M:%S", mod_time)
    print(time_str + ": Last change to " + str(path_to_file))


def plot_prec_rec_local(data, result_name: str):
    df_clean = helps.load_original_data(data, load_dirty=False)
    df_dirty = helps.load_original_data(data, load_dirty=True)

    results = load_result(Path(f"{data.results_path}{result_name}.p"))
    local_results = list(filter(lambda x: x.get('label'), results))
    prec, rec = {}, {}
    for r in local_results:
        df_clean_y_true = df_clean.loc[:, r['label']]
        imputer = datawig.AutoGluonImputer.load(output_path='./',
                                                model_name=r['model_checksum'])

        if imputer.predictor.problem_type in ('multiclass', 'binary'):
            df_probas = imputer.predict(df_dirty, return_probas=True)

            # pu.figure_setup()
            # fig_size = pu.get_fig_size(10, 4)
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            for i in imputer.predictor.class_labels:
                prec[i], rec[i], _ = precision_recall_curve(df_clean_y_true == i,
                                                            df_probas.loc[:, i],
                                                            pos_label=True)
                ax.plot(rec[i], prec[i], label=f'class {i}')

            plt.legend(loc="best")
            ax.set(xlabel='Recall', ylabel='Precision')
            ax.set_title("Data Cleaning Performance Column "
                         f"{data.column_map[r['label']]}")
            ax.set_axisbelow(True)
            plt.tight_layout()
            # p = Path(data.figures_path + result_name)
            # p.mkdir(parents=True, exist_ok=True)
            # fig.savefig(p + result_name + '.png', dpi=300, tight=True)
            # print(f'Successfully saved the figure to {data.figures_path}.')

            # plt.title("Precision - Recall Curve")
            plt.show()


def plot_prec_threshold(data, *args):
    global_results, prec_thresh = list(), list()

    p = Path(data.results_path)
    for r_path in p.glob('*.p'):
        result = load_result(r_path)
        glob = list(filter(lambda x: x.get('global_error_detection'), result))
        global_results.append(glob)

    for r_path in p.glob('*.p'):
        result = load_result(r_path)
        conf = result[0].get('precision_threshold')
        prec_thresh.append(conf)

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 4)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    clean = [r[0]['global_error_cleaning'] for r in global_results]
    detect = [r[0]['global_error_detection'] for r in global_results]
    ax.scatter(prec_thresh,
               clean,
               label='Error Cleaning Complete Dataset')
    ax.scatter(prec_thresh,
               detect,
               label='Error Detecting Complete Dataset')
    ax.legend()

    ax.set_title('Effect Of Precision Threshold on Cleaning and Detection Performance')
    ax.set(xlabel='Precision Threshold',
           ylabel='F1 Score')
    return(fig, ax)



def plot_auc_cleaning_global(data, *args):
    """
    Plot the trend of cleaning models over time using auc
    of the precision-recall curve.
    """
    df_clean = helps.load_original_data(data, load_dirty=False)
    df_dirty = helps.load_original_data(data, load_dirty=True)

    local_results, timestamps = list(), list()

    p = Path(data.results_path)
    for r_path in p.glob('*.p'):
        result = load_result(r_path)
        local_result = list(filter(lambda x: x.get('label'), result))
        ts = list(map(lambda x: x.get('run_at_timestamp'), filter(lambda x: x.get('run_at_timestamp'), result)))

        local_results.append(local_result)
        timestamps.append(ts)

    prec, rec = {}, {}
    avg_areas_under_curve_per_run, aucs = list(), list()

    for local_result in local_results:  # for each cleaning run
        for r in local_result:  # for each RHS
            aucs = list()
            df_clean_y_true = df_clean.loc[:, r['label']]
            imputer = datawig.AutoGluonImputer.load(output_path='./',
                                                    model_name=r['model_checksum'])

            df_probas = imputer.predict(df_dirty, return_probas=True)
            for i in imputer.predictor.class_labels:  # for each class
                prec[i], rec[i], _ = precision_recall_curve(df_clean_y_true == i,
                                                            df_probas.loc[:, i],
                                                            pos_label=True)
                aucs.append(auc(rec[i], prec[i]))
        avg_areas_under_curve_per_run.append(np.average(aucs))
    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 4)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)
    ax.scatter(timestamps, avg_areas_under_curve_per_run, label='AUC Cleaning')
    ax.legend()

    ax.set(xlabel='Timestamp',
           ylabel='AUC Cleaning Performance')
    return(fig, ax)


def plot_f1_cleaning_detection_global(data, *kwargs):
    """
    Plot the trend of cleaning models over time.
    """
    p = Path(data.results_path)
    all_results = []
    for r_path in p.glob('*.p'):
        all_results = all_results + load_result(r_path)
    global_results = list(filter(lambda x: x.get('global_error_detection'),
                          all_results))
    timestamps = list(map(lambda x: x.get('run_at_timestamp'), filter(lambda x: x.get('run_at_timestamp'), all_results)))

    detection = [x['global_error_detection'] for x in global_results]
    cleaning = [x['global_error_cleaning'] for x in global_results]
    prec_thresholds = [x['precision_threshold'] for x in all_results
                       if x.get('precision_threshold') is not None]

    print("Plotting Datapoints:")
    for x in zip(detection, cleaning, timestamps, prec_thresholds):
        print('~~~~~')
        print(f'Detection performance: {round(x[0], 5)}')
        print(f'Cleaning performance: {round(x[1], 5)}')
        print(f'Precision Threshold: {x[3]}')
        print(f'Timestamp: {x[2]}')

    pu.figure_setup()
    fig_size = pu.get_fig_size(10, 4)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)
    ax.scatter(timestamps, detection, label='F1 Error Detection')
    ax.scatter(timestamps, cleaning, label='F1 Data Cleaning')
    ax.legend()

    ax.set(xlabel='Timestamp',
           ylabel='Cleaning Performance')
    return(fig, ax)


def plot_f1_cleaning_detection_local(data, result_name: str):
    """
    Plot the result stored at $data's $results_path with the name
    $result_name.
    """
    results = load_result(Path(f"{data.results_path}{result_name}.p"))

    local_results = list(filter(lambda x: x.get('label'), results))

    labels = [data.column_map[c['label']] for c in local_results]
    perf_error_detection = [round(c['error_detection'], 2)
                            for c in local_results]
    perf_cleaning = [round(c['error_cleaning'], 2)
                     for c in local_results]

    pu.figure_setup()
    fig_size = pu.get_fig_size(25, 5)
    fig = plt.figure(figsize=list(fig_size))
    ax = fig.add_subplot(111)

    x = np.arange(len(labels))
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width/2, perf_cleaning,
                    width, label='Cleaning')
    rects2 = ax.bar(x + width/2, perf_error_detection,
                    width, label='Error Detection')

    ax.set_ylabel('F1-Score')
    ax.set_title('Performance Cleaning \& Error Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    return(fig, ax)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=main.__doc__)
    # 'Script for plotting figures for Philipp Jung\'s Master Thesis.\n')

    parser.add_argument('-s',
                        '--save',
                        help='specify filename and -type to save the figure.')
    parser.add_argument('-f', '--figure', help='specify a figure to plot.')
    parser.add_argument('-d', '--data',
                        help='specify a dataset to use results of.')
    parser.add_argument('-n', '--name', help='name of the result file.')
    parser.add_argument('-a', '--all',
                        help='plot and save all possible plots.',
                        action='store_true')

    args = parser.parse_args()
    main(args)
