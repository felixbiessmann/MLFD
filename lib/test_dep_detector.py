import unittest
import lib.dep_detector as dep
import lib.constants as c
import anytree as tree
import lib.test_data.test_trees as test_trees


class TestDepDetector(unittest.TestCase):

    def setUp(self):
        self.Detector = dep.DepOptimizer(c.NURSERY)

    def test_RootNode(self):
        r = dep.RootNode(name='test_root_node',
                         columns=[],
                         train='',
                         validate='',
                         test='',
                         continuous=[],
                         is_continuous=True)
        self.assertIsNone(r.score)
        self.assertNotEqual(r.get_newest_children(), ())

    def test_load_data_and_init_roots(self):
        self.Detector.load_data()
        self.Detector.init_roots()

        # load_data
        self.assertIsNotNone(self.Detector.df_train)
        self.assertIsNotNone(self.Detector.df_validate)
        self.assertIsNotNone(self.Detector.df_test)
        self.assertIsInstance(self.Detector.columns, list)

        # init_roots
        self.assertEqual(len(self.Detector.columns),
                         len(self.Detector.roots))
        for root in self.Detector.roots.values():
            self.assertIsNotNone(root.score)
            self.assertNotEqual(root.get_newest_children(),
                                ())

    def test_get_initial_children(self):
        self.Detector.load_data()
        self.Detector.init_roots()
        for root in self.Detector.roots.values():
            root.generate_scores()
            initial_children = root.get_newest_children()
            self.assertEqual(1, len(initial_children))
            self.assertNotIn(root, initial_children[0].name)
            self.assertIsNotNone(initial_children[0].score)

    def test_search_dependencies(self):
        self.Detector.search_dependencies('complete', True)

        for root in self.Detector.roots.values():
            self.assertIsNotNone(root.get_newest_children()[0].score)

    def test_run_top_down(self):
        self.Detector.load_data()
        self.Detector.init_roots()

        max_depth = 0
        for root in self.Detector.roots.values():
            root.run_top_down('complete', True)
            node_depth = root.get_newest_children()[0].depth
            if max_depth < node_depth:
                max_depth = node_depth

        reasonable_depth = len(self.Detector.columns) / 6
        self.assertTrue(max_depth > reasonable_depth)

    def test_get_greedy_candidates(self):
        self.Detector.load_data()
        self.Detector.init_roots()
        root = self.Detector.roots[9]
        root.children[0].score = 1.0
        root.get_greedy_candidates()
        no_roots = len(self.Detector.roots.values())
        self.assertEqual(len(root.get_newest_children()),
                         no_roots-1)
        for child in root.children[0].children:
            child.score = 1.0
        root.get_greedy_candidates()
        self.assertEqual(len(root.get_newest_children()),
                         no_roots-2)

    def test_get_complete_candidates(self):
        self.Detector.load_data()
        self.Detector.init_roots()

        root = self.Detector.roots[9]
        root.children[0].score = 1.0
        root.get_complete_candidates()
        no_roots = len(self.Detector.roots.values())
        self.assertEqual(len(root.get_newest_children()),
                         no_roots-1)
        for child in root.get_newest_children():
            child.score = 1.0
        root.get_complete_candidates()
        self.assertEqual(len(root.get_newest_children()),
                         (no_roots-1) * (no_roots-2))

    def test_get_min_dep(self):
        self.Detector.search_dependencies('complete', dry_run=True)
        self.Detector.get_minimal_dependencies()

    def test_get_continuous_min_dep_greedy(self):
        root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                            is_continuous=True, threshold=10)
        root.search_strategy = 'greedy'
        d, c = test_trees.get_continuous_greedy_tree()
        self.assertEqual(dep.get_continuous_min_dep(root),
                         {tuple(d.name): d.score})
        d.score = 99
        self.assertEqual(dep.get_continuous_min_dep(root),
                         {tuple(c.name): c.score})

    def test_get_continuous_min_dep_complete(self):
        root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                            is_continuous=True, threshold=10)
        root.search_strategy = 'complete'
        a = root.children[0]
        a.score = 9
        b = tree.Node([1, 2], score=8, parent=a)
        c = tree.Node([2, 3], score=8, parent=a)
        d = tree.Node([2], score=1, parent=c)
        e = tree.Node([3], score=99, parent=c)
        self.assertEqual(dep.get_continuous_min_dep(root),
                         {tuple(b.name): b.score,
                          tuple(d.name): d.score})

    def test_get_non_continuous_min_dep_greedy(self):
        root = dep.RootNode(0, [0, 1, 2, 3], '', '', '', [],
                            is_continuous=False, threshold=0.9)
        root.search_strategy = 'greedy'
        a = root.children[0]
        a.score = 0.94
        b = tree.Node([1, 2], score=0.93, parent=a)
        c = tree.Node([2, 3], score=0.93, parent=a)
        d = tree.Node([2], score=0.80, parent=c)
        e = tree.Node([3], score=1, parent=c)

        self.assertEqual(dep.get_non_continuous_min_dep(root),
                         {tuple(e.name): e.score})
        d.score = 0.92
        self.assertEqual(dep.get_non_continuous_min_dep(root),
                         {tuple(e.name): e.score,
                          tuple(d.name): d.score})


if __name__ == '__main__':
    unittest.main()
