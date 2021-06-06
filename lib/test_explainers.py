import numpy as np
from lib.explainers import (
    get_n_best_features
)


def test_get_n_best_features():
    global_shaps = np.array([0.1, 0.5, 1, 7, 8])
    n = 5
    assert get_n_best_features(n, global_shaps) == [4, 3, 2, 1, 0]
