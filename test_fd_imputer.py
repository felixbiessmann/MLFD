import unittest
import pandas as pd
import fd_imputer


class TestFdImputer(unittest.TestCase):

    def setUp(self):
        # make up test data
        self.df_test = pd.DataFrame(
                [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']],
                columns=[1, 2, 3])
        self.df_train = pd.DataFrame(
                [['d', 'e', 'f'], ['g', 'h', 'i'], ['a', 'b', 'c']],
                columns=[1, 2, 3])
        self.fd = {3: [2, 1]}
        self.lhs = [2, 1]

    def test_select_LHS_row(self):
        self.assertEqual(fd_imputer.select_LHS_row(
            self.df_train.iloc[0], self.df_test, self.lhs).index[0],
            1)


if __name__ == '__main__':
    unittest.main()
