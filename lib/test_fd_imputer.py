import os
import itertools
import unittest
import pandas as pd
import numpy as np
import lib.fd_imputer as fd
import tempfile
import random


class TestFdImputer(unittest.TestCase):

    def setUp(self):
        # set up test data
        self.data_title = 'test'
        self.continuous_cols = [0, 1, 2, 3]
        self.fd = {3: [2, 1]}
        self.df_train = pd.DataFrame([[1, 2, 3], [7, 8, 15], [11, 13, 24],
                                      [np.nan, np.nan, np.nan]],
                                     columns=[0, 1, 2])
        self.df_validate = pd.DataFrame([[1, 2, 3], [4, 5, 9], [5, 7, 12]],
                                        columns=[0, 1, 2])
        self.df_test = pd.DataFrame([[8, 9, 17], [10, 10, 20], [20, 30, 50]],
                                    columns=[0, 1, 2])
        self.test_df = pd.concat([self.df_train,
                                  self.df_validate,
                                  self.df_test],
                                 ignore_index=True)

        # distance to A, decimal code for ASCII characters, character as str
        self.mixed_train = pd.DataFrame([[0.0, 65, 'A'], [1.0, 66, 'B'],
                                         [2.0, 67, 'C'], [5.0, 70, 'F'],
                                         [7.0, 72, 'H'], [8.0, 73, 'I']],
                                        columns=[0, 1, 2])
        self.mixed_validate = pd.DataFrame([[3.0, 68, 'D'], [4.0, 69, 'E'],
                                            [6.0, 71, 'G'], [7.5, 72, 'H'],
                                            [-3.2, 73, 'I']],
                                           columns=[0, 1, 2])
        self.mixed_fd = {2: [3]}
        self.mixed_continuous = []

        # real data from production for testing
        self.fds = fd.read_fds('lib/test_data/test_fd.txt')

    def test_get_performance(self):
        """ run test on adult.csv data """
        df_train, df_validate, df_test = fd.load_dataframes(
            './MLFD_fd_detection/data/',
            'adult',
            'noValueSetHere123156456')
        df_train = df_train.loc[:2500, :]
        df_validate = df_validate.loc[:2500, :]
        df_test = df_test.loc[0:2500, :]
        fds = {4: [5], 5: [4]}  # these FDs are symmetric
        sym_result = []
        for rhs in fds:
            lhs = fds[rhs]
            df_imputed = fd.fd_imputer(df_train, df_validate, {rhs: lhs})
            result = fd.get_performance(df_imputed, rhs, lhs, [])
            sym_result.append(result)
        print('FD Imputer performance results: {}'.format(str(sym_result)))

        self.assertTrue(sym_result[0], sym_result[1])

        ml_sym_result = []
        for rhs in fds:
            df_train = df_train.astype('str')
            df_validate = df_validate.astype('str')
            df_test = df_test.astype('str')
            ml_df_imputed = fd.ml_imputer(df_train, df_train, df_train, str(rhs))
            ml_sym_result.append(fd.get_performance(ml_df_imputed, rhs, lhs, []))
        print('ML performance results: {}'.format(str(ml_sym_result)))
        mixed_fd = self.mixed_fd
        rhs = list(mixed_fd.keys())[0]
        lhs = mixed_fd[rhs]
        continuous = self.mixed_continuous
        df_train, df_validate = (map(fd.index_as_first_column,
                                     [self.mixed_train,
                                      self.mixed_validate]))
        df_imputed = fd.fd_imputer(df_train, df_validate, mixed_fd)
        result = fd.get_performance(df_imputed, str(rhs), str(lhs), continuous)
        print(result)

    def test_random_dependency_generator(self):
        n = random.randint(0, 800)
        cols = list(range(0, 30))

        dependencies = fd.random_dependency_generator(cols, n)
        for rhs in dependencies:
            lhs = dependencies[rhs]
            lhs.sort()
            no_lhs = len(lhs)
            no_unique_lhs = len(list(lhs for lhs, _ in itertools.groupby(lhs)))
            self.assertEqual(no_lhs, no_unique_lhs)
        no_dependencies = sum([len(dependencies[rhs]) for rhs in dependencies])
        self.assertTrue(no_dependencies <= n)

    def test_run_ml_imputer_on_fd_set(self):
        fds = self.fd
        rhs = list(fds.keys())[0]
        lhs = fds[rhs]
        fds[rhs] = [fds[rhs]]

        df_train, df_validate, df_test = (map(fd.index_as_first_column,
                                              [self.df_train,
                                               self.df_validate,
                                               self.df_test]))
        ml_imputer_results = fd.run_ml_imputer_on_fd_set(
            df_train,
            df_validate,
            df_test,
            fds,
            self.continuous_cols)

        self.assertIn(3, ml_imputer_results)
        self.assertIn('lhs', ml_imputer_results[rhs][0])
        self.assertEqual(ml_imputer_results[rhs][0]['lhs'],
                         list(map(str, lhs)))

    def test_run_fd_imputer_on_fd_set(self):
        df_train, df_validate = (map(fd.index_as_first_column,
                                     [self.df_train,
                                      self.df_validate]))
        fds = self.fd
        fds[3] = [fds[3]]
        result = fd.run_fd_imputer_on_fd_set(df_train,
                                             df_validate,
                                             self.fd,
                                             self.continuous_cols)

        self.assertEqual(len(result.keys()), 1)
        self.assertEqual(list(result.keys())[0], 3)
        self.assertEqual(2, result[3][0]['nans'])

    def test_ml_imputer(self):
        df_train, df_validate, df_test = fd.split_df(
            'adult', self.test_df, [0.6, 0.2, 0.2])
        df_imputed = fd.ml_imputer(df_train,
                                   df_validate,
                                   df_test,
                                   str(3))

        self.assertTrue(df_imputed.shape[1], self.test_df.shape[1]+1)
        self.assertTrue(df_test.shape[0], df_imputed.shape[0])

    def test_split_df(self):
        # does split_df() set the index as the 0th column?
        df_train, df_validate, df_test = fd.split_df(
            'adult', self.test_df, [0.6, 0.2, 0.2])

        train_index = np.array(df_train.index)
        train_index_col = np.array(df_train.iloc[:, 0].values)
        self.assertTrue(np.array_equal(train_index, train_index_col))

        # does split_df() recognize if the index is already a column?
        df_train, df_validate, df_test = fd.split_df(
            'adult', self.test_df.reset_index(), [0.6, 0.2, 0.2])

        no_train_cols = len(df_train.columns)
        no_original_cols = len(self.test_df.reset_index().columns)
        self.assertTrue(no_train_cols == no_original_cols)

    def test_read_fds(self):
        rhs = [x for x in self.fds]
        self.assertTrue(min(rhs) == 1)  # index is not a rhs
        self.assertTrue(max(rhs) == (len(rhs)))  # same reason as above

        lhs_flat = [z for x in self.fds for y in self.fds[x] for z in y]
        self.assertTrue(min(lhs_flat) == 0)
        self.assertTrue(max(lhs_flat) == (len(rhs)))

    def test_split_load_df(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            fd.split_df(
                self.data_title, self.test_df, [0.33, 0.33, 0.33],
                tempdirname+'/')

            df_train, df_validate, df_test = fd.load_dataframes(
                tempdirname+'/', self.data_title, 'noData')

            df_glued = pd.concat([df_train, df_validate, df_test])
            self.assertEqual(sum(df_glued.duplicated()), 0)

            # check if missing value deletion from load_dataframes works
            self.assertTrue(df_glued.isna().values.sum() == 0)

            self.assertTrue(df_glued.shape[0] <= self.test_df.shape[0])

    def test_fd_imputer(self):
        df_train, df_validate = (map(fd.index_as_first_column,
                                     [self.df_train,
                                      self.df_validate]))
        df_mixed_train, df_mixed_validate = (map(fd.index_as_first_column,
                                                 [self.mixed_train,
                                                  self.mixed_validate]))

        self.assertEqual(fd.fd_imputer(
            df_train, df_validate, self.fd).index[0],
            0)
        self.assertEqual(fd.fd_imputer(
            df_train,
            df_validate,
            self.fd).loc[:, '3_imputed'].dropna()[0],
            3)
        self.assertEqual(len(fd.fd_imputer(df_mixed_train,
                                           df_mixed_validate,
                                           self.mixed_fd).loc[:, '2_imputed'].dropna()),
                         2)


if __name__ == '__main__':
    unittest.main()
