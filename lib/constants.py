import os
root = os.path.dirname(os.path.abspath(__file__))+'/..'

METANOME_DATA_PATH = f'{root}/MLFD_fd_detection/backend/WEB-INF/classes/inputData/'


class Dataset:
    def __init__(self, title, fd_path, continuous,
                 original_separator, missing_value_token, origin):
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
        self._origin = origin


ABALONE = Dataset(title='abalone',
                  original_separator=',',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084553_fds',
                  continuous=[2, 3, 4, 5, 6, 7, 8],
                  missing_value_token='',
                  origin='https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')

ADULT = Dataset(title='adult',
                fd_path='HyFD-1.2-SNAPSHOT.jar2019-07-06T091216_fds',
                continuous=[1, 3, 11, 12, 13],
                original_separator=', ',
                missing_value_token='noValueSetHere123156456',
                origin='https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')

BALANCESCALE = Dataset(title='balance-scale',
                       original_separator=',',
                       fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084712_fds',
                       continuous=[0],
                       missing_value_token='',
                       origin='https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data')

BREASTCANCER = Dataset(title='breast-cancer',
                       original_separator=',',
                       fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084747_fds',
                       continuous=[1],
                       missing_value_token='?',
                       origin='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')

BRIDGES = Dataset(title='bridges',
                  original_separator=',',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T090753_fds',
                  continuous=[0],
                  missing_value_token='?',
                  origin='https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version2')

CERVICAL_CANCER = Dataset(title='cervical-cancer',
                          original_separator=',',
                          fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T084747_fds',
                          continuous=[0, 1, 2, 3, 8, 10, 12, 25, 26, 27],
                          missing_value_token='?',
                          origin='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data')

ECHOCARDIOGRAM = Dataset(title='echocardiogram',
                         original_separator=',',
                         fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085015_fds',
                         continuous=[1, 3, 5, 6, 7, 8, 9, 10],
                         missing_value_token='?',
                         origin='https://archive.ics.uci.edu/ml/machine-learning-databases/echocardiogram/echocardiogram.data')

HEPATITIS = Dataset(title='hepatitis',
                    original_separator=',',
                    fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T090716_fds',
                    continuous=[15, 18],
                    missing_value_token='?',
                    origin='https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data')

HORSE = Dataset(title='horse',
                original_separator=' ',
                fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085141_fds',
                continuous=[3, 4, 5, 6, 16, 19, 20, 22],
                missing_value_token='?',
                origin='https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data')

IRIS = Dataset(title='iris',
               original_separator=',',
               fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085225_fds',
               continuous=[1, 2, 3, 4],
               missing_value_token='',
               origin='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

LETTER = Dataset(title='letter',
                 original_separator=',',
                 fd_path='HyFD-1.2-SNAPSHOT.jar2019-08-25T085300_fds',
                 continuous=[0],  # this is hard to decide
                 missing_value_token='',
                 origin='https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data')

NURSERY = Dataset(title='nursery',
                  fd_path='HyFD-1.2-SNAPSHOT.jar2019-07-04T182440_fds',
                  continuous=[],
                  original_separator=',',
                  missing_value_token='',
                  origin='https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data')

