import unittest
import lib.dep_detector as dep
import lib.constants as c


class TestDepDetector(unittest.TestCase):

    def setUp(self):
        self.Detector = dep.DepOptimizer(c.NURSERY)

    def test_RootNode(self):
        r = dep.RootNode(name='test_root_node', is_continuous=True)
        self.assertIsNone(r.score)
        self.assertIsNone(r.threshold)
        self.assertEqual(r.get_newest_children(), ())

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
            self.assertIsNone(root.score)
            self.assertEqual(root.get_newest_children(),
                             ())

    def test_get_initial_children(self):
        self.Detector.load_data()
        self.Detector.init_roots()
        self.Detector.get_top_down_candidates()
        for root in self.Detector.roots.values():
            initial_children = root.get_newest_children()
            self.assertEqual(1, len(initial_children))
            self.assertNotIn(root, initial_children[0].name)
            self.assertIsNone(initial_children[0].score)

    def test_get_children(self):
        pass

if __name__ == '__main__':
    unittest.main()
