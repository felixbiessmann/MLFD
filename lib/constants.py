import os
root = os.path.dirname(os.path.abspath(__file__))+'/..'

METANOME_DATA_PATH = f'{root}/MLFD_fd_detection/backend/WEB-INF/classes/inputData/'


class Dataset:
    def __init__(self, title, fd_path, continuous,
                 original_separator, missing_value_token):
        self.title = title
        self.figures_path = f'{root}/figures/{title}/'
        self.results_path = f'{root}/figures/results/{title}/'
        self.fd_path = f'{root}/MLFD_fd_detection/results/{fd_path}'
        self.continuous = continuous
        self.random_fd_path = f'{self.results_path}random_fds.p'
        self.data_path = f'{root}/MLFD_fd_detection/data/{title}.csv'
        self.splits_path = f'{root}/MLFD_fd_detection/data/'
        self.original_separator = original_separator
        self.missing_value_token = missing_value_token
        self.dep_optimizer_results_path = f'{root}/figures/results/{title}'


ADULT = Dataset(title='adult',
                fd_path='HyFD-1.2-SNAPSHOT.jar2019-07-06T091216_fds',
                continuous=[0, 1, 3, 11, 12, 13],
                original_separator=';',
                missing_value_token='noValueSetHere123156456')

NURSERY = Dataset(title='nursery',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-07-04T182440_fds',
                  continuous=[0],
                  original_separator=',',
                  missing_value_token='')

ABALONE = Dataset(title='abalone',
                  original_separator=',',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084553_fds',
                  continuous=[0, 2, 3, 4, 5, 6, 7, 8],
                  missing_value_token='')

BALANCESCALE = Dataset(title='balance-scale',
                       original_separator=',',
                       fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084712_fds',
                       continuous=[0],
                       missing_value_token='')

BREASTCANCER = Dataset(title='breast-cancer-wisconsin',
                       original_separator=',',
                       fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084747_fds',
                       continuous=[0, 1],
                       missing_value_token='?')

BRIDGES = Dataset(title='bridges',
                  original_separator=',',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T090753_fds',
                  continuous=[0],
                  missing_value_token='?')

CHESS = Dataset(title='chess',
                original_separator=',',
                fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084928_fds',
                continuous=[0],
                missing_value_token='')

ECHOCARDIOGRAM = Dataset(title='echocardiogram',
                         original_separator=',',
                         fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085015_fds',
                         continuous=[0, 1, 3, 5, 6, 7, 8, 9, 10],
                         missing_value_token='?')

HEPATITIS = Dataset(title='hepatitis',
                    original_separator=',',
                    fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T090716_fds',
                    continuous=[0, 15, 18],
                    missing_value_token='?')

HORSE = Dataset(title='horse',
                original_separator=';',
                fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085141_fds',
                continuous=[0, 3, 4, 5, 6, 16, 19, 20, 22],
                missing_value_token='?')

IRIS = Dataset(title='iris',
               original_separator=',',
               fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085225_fds',
               continuous=[0, 1, 2, 3, 4],
               missing_value_token='')

LETTER = Dataset(title='letter',
                 original_separator=',',
                 fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085300_fds',
                 continuous=[0],  # this is hard to decide
                 missing_value_token='')

MOVIES_DURATION = Dataset(title='movies_duration',
                         original_separator=',',
                         fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085300_fds',
                         continuous=[0, 1],
                         missing_value_token='')

MOVIES_ORDERING = Dataset(title='movies_ordering',
                         original_separator=',',
                         fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085300_fds',
                         continuous=[0, 1],
                         missing_value_token='')
