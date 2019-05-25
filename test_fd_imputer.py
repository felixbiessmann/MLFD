import unittest
import pandas as pd
import fd_imputer


class TestFdImputer(unittest.TestCase):

    def setUp(self):
        # make up test data
        self.df_test = pd.DataFrame([[1, 2, 3], [4, 5, 9], [5, 7, 12]],
                                    columns=[0, 1, 2])
        self.df_train = pd.DataFrame([[1, 2, 4], [7, 8, 15], [11, 13, 24]],
                                     columns=[0, 1, 2])
        self.fd = {2: [1, 0]}

    def test_fd_imputer(self):
        self.assertEqual(fd_imputer.fd_imputer(
            self.df_test, self.df_train, self.fd).index[0],
            0)
        self.assertEqual(fd_imputer.fd_imputer(
            self.df_test, self.df_train, self.fd).iloc[0, 2],
            4)


if __name__ == '__main__':
    unittest.main()
