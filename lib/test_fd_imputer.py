import unittest
import pandas as pd
import fd_imputer
import tempfile


class TestFdImputer(unittest.TestCase):

    def setUp(self):
        # make up test data
        self.df_test = pd.DataFrame([[1, 2, 3], [4, 5, 9], [5, 7, 12]],
                                    columns=[0, 1, 2])
        self.df_train = pd.DataFrame([[1, 2, 4], [7, 8, 15], [11, 13, 24]],
                                     columns=[0, 1, 2])
        self.fd = {2: [1, 0]}

        self.split_df = pd.DataFrame([[1, 2, 3], [2, 1, 3], [5, 6, 7],
                                      [3, 1, 5], [4, 3, 1], [4, 1, 7]])
        self.fds = fd_imputer.read_fds('test_data/test_fd.txt')

    def test_read_fds(self):
        rhs = [x for x in self.fds]
        self.assertTrue(min(rhs) == 1)  # index is not a rhs
        self.assertTrue(max(rhs) == (len(rhs)))  # same reason as above

        lhs_flat = [z for x in self.fds for y in self.fds[x] for z in y]
        self.assertTrue(min(lhs_flat) == 0)
        self.assertTrue(max(lhs_flat) == (len(rhs)))

    def test_df_split(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            fd_imputer.df_split(
                'test', self.split_df, [0.33, 0.33, 0.33], tempdirname+'/')

            df_train = pd.read_csv(
                tempdirname+'/train/test_train.csv', header=None)
            df_validate = pd.read_csv(
                tempdirname+'/validate/test_validate.csv', header=None)
            df_test = pd.read_csv(
                tempdirname+'/test/test_test.csv', header=None)

            no_duplicates = sum(
                pd.concat([df_test, df_train, df_validate]).duplicated())
            self.assertEqual(no_duplicates, 0)

    def test_fd_imputer(self):
        self.assertEqual(fd_imputer.fd_imputer(
            self.df_test, self.df_train, self.fd).index[0],
            0)
        self.assertEqual(fd_imputer.fd_imputer(
            self.df_test,
            self.df_train,
            self.fd).loc[:, '2_imputed'].dropna()[0],
            4)


if __name__ == '__main__':
    unittest.main()
