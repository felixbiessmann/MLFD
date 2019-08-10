import os
root = os.path.dirname(os.path.abspath(__file__))+'/..'

METANOME_DATA_PATH = root+'/MLFD_fd_detection/backend/WEB-INF/classes/inputData/'


class Dataset:
    def __init__(self, title, figures_path, results_path, fd_path, continuous,
                 random_fd_path, original_separator, missing_value_token):
        self.title = title
        self.figures_path = figures_path
        self.results_path = results_path
        self.fd_path = fd_path
        self.continuous = continuous
        self.random_fd_path = random_fd_path
        self.data_path = root+'/MLFD_fd_detection/data/'+title+'.csv'
        self.splits_path = root+'/MLFD_fd_detection/data/'
        self.original_separator = original_separator
        self.missing_value_token = missing_value_token
        self.dep_optimizer_results_path = root+'/figures/results/'\
            + title+'/'


ADULT = Dataset(title='adult',
                figures_path=root+'/figures/adult/',
                results_path=root+'/figures/results/adult/',
                fd_path=root+'/MLFD_fd_detection/results/HyFD-1.2-SNAPSHOT.jar2019-07-06T091216_fds',
                continuous=[0, 1, 3, 11, 12, 13],
                random_fd_path=root+'/figures/results/adult/random_fds.p',
                original_separator=';',
                missing_value_token='noValueSetHere123156456')

NURSERY = Dataset(title='nursery',
                  figures_path=root+'/figures/nursery/',
                  results_path=root+'/figures/results/nursery/',
                  fd_path=root+'/MLFD_fd_detection/results/HyFD-1.2-SNAPSHOT.jar2019-07-04T182440_fds',
                  continuous=[0],
                  random_fd_path=root+'/figures/results/nursery/random_fds.p',
                  original_separator=',',
                  missing_value_token='')
