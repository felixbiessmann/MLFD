import anytree as tree
import lib.dep_detector as dep_detector
import lib.dep_detector as dep


def get_continuous_greedy_tree():
    """
    Create a tree for a specific test case and return the nodes that
    the assertion depends on.
    """
    root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                        is_continuous=True, threshold=10)
    root.search_strategy = 'greedy'
    a = root.children[0]
    a.score = 9
    b = tree.Node([1, 2], score=99, parent=a)
    c = tree.Node([2, 3], score=8, parent=a)
    d = tree.Node([2], score=1, parent=c)
    e = tree.Node([3], score=99, parent=c)

    return (root, d, c)


def get_continuous_complete_tree():
    root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                        is_continuous=True, threshold=10)
    root.search_strategy = 'complete'
    a = root.children[0]
    a.score = 9
    b = tree.Node([1, 2], score=8, parent=a)
    c = tree.Node([2, 3], score=8, parent=a)
    d = tree.Node([2], score=1, parent=c)
    e = tree.Node([3], score=99, parent=c)

    return (root, b, d)


def get_non_continuous_greedy_tree():
    root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                        is_continuous=False, threshold=0.9)
    root.search_strategy = 'greedy'
    a = root.children[0]
    a.score = 0.94
    b = tree.Node([1, 2], score=0.93, parent=a)
    c = tree.Node([2, 3], score=0.93, parent=a)
    d = tree.Node([2], score=0.80, parent=c)
    e = tree.Node([3], score=1, parent=c)

    return (root, e, d)
