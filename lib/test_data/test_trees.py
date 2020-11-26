import anytree as tree
import lib.dep_detector as dep_detector


def get_continuous_greedy_tree():
    root = dep_detector.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                        is_continuous=True, threshold=10)
    root.search_strategy = 'greedy'
    a = root.children[0]
    a.score = 9
    b = tree.Node([1, 2], score=99, parent=a)
    c = tree.Node([2, 3], score=8, parent=a)
    d = tree.Node([2], score=1, parent=c)
    e = tree.Node([3], score=99, parent=c)

    return (d, c)
