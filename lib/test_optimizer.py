import pytest
import lib.constants as c
from lib.optimizer import (
    DepOptimizer,
    RootNode,
    run_binary_search
)
import pandas as pd


@pytest.fixture
def Detector():
    return DepOptimizer(c.NURSERY)


def test_RootNode():
    r = RootNode(name='test_root_node',
                 columns=[],
                 train='',
                 validate='',
                 test='',
                 continuous=[],
                 is_continuous=True)
    assert r.score is None
    assert r.get_newest_children() != ()


def test_load_data_and_init_roots(Detector):
    Detector.load_data()
    Detector.init_roots()

    # load_data
    assert Detector.df_train is not None
    assert Detector.df_validate is not None
    assert Detector.df_test is not None
    assert isinstance(Detector.columns, list)

    # init_roots
    assert len(Detector.columns) == len(Detector.roots)
    for root in Detector.roots.values():
        assert root.score is not None
        assert root.get_newest_children() != ()


def test_get_initial_children(Detector):
    Detector.load_data()
    Detector.init_roots()
    for root in Detector.roots.values():
        root.generate_scores()
        initial_children = root.get_newest_children()
        assert 1 == len(initial_children)
        assert root not in initial_children[0].name
        assert initial_children[0].score is not None


def test_search_dependencies(Detector):
    Detector.search_dependencies('complete', True)

    for root in Detector.roots.values():
        assert root.get_newest_children()[0].score is not None


def test_run_top_down(Detector):
    Detector.load_data()
    Detector.init_roots()

    max_depth = 0
    for root in Detector.roots.values():
        root.run_top_down('complete', True)
        node_depth = root.get_newest_children()[0].depth
        if max_depth < node_depth:
            max_depth = node_depth

    reasonable_depth = len(Detector.columns) / 6
    assert max_depth > reasonable_depth


def test_get_greedy_candidates(Detector):
    Detector.load_data()
    Detector.init_roots()
    root = Detector.roots[8]
    root.children[0].score = 1.0
    root.get_greedy_candidates()
    no_roots = len(Detector.roots.values())
    assert len(root.get_newest_children()) == no_roots-1
    for child in root.children[0].children:
        child.score = 1.0
    root.get_greedy_candidates()
    assert len(root.get_newest_children()) == no_roots-2


def test_get_complete_candidates(Detector):
    Detector.load_data()
    Detector.init_roots()

    root = Detector.roots[8]
    root.children[0].score = 1.0
    root.get_complete_candidates()
    no_roots = len(Detector.roots.values())
    assert len(root.get_newest_children()) == no_roots-1
    for child in root.get_newest_children():
        child.score = 1.0
    root.get_complete_candidates()
    assert len(root.get_newest_children()) == (no_roots-1) * (no_roots-2)


def test_get_min_dep(Detector):
    Detector.search_dependencies('complete', dry_run=True)
    Detector.get_minimal_dependencies()


def test_run_binary_search():
    se_importance = pd.Series([0.05, 0.00, 0.00, 0.21, 0.11, 0.09, 0.20, 0.30, 0.05])
    se_importance = se_importance.sort_values()
    p_fun = lambda se, lhs_limit: sum(se.iloc[:lhs_limit])
    opt_lhs = run_binary_search(se_importance, 0.3, p_fun)
    assert opt_lhs == [0, 1, 2, 4, 5, 8]
