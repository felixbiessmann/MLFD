import lib.fd_imputer as fd
import anytree as tree
import pandas as pd
import numpy as np
import random


class RootNode(tree.NodeMixin):
    """
    Root node class for dependency search-trees.

    Methods:
    get_newest_children -- Returns node(s) that is/are the most recent child
        or children of a tree

    Attributes:
    name -- potential rhs column name
    score -- empty string, to be homogenous with other Nodes
    is_continuous -- boolean, True if column contains continuous values, False
    if values are classifiable
    threshold - threshold that child-nodes are compared to when searching for
    dependencies. Either MSE or f1-score, depending on is_continuous.
    """

    def __init__(self, name, is_continuous: bool, threshold=None):
        self.name = name
        self.score = None
        self.is_continuous = is_continuous
        self.threshold = threshold

    def __str__(self):
        return(str(self.name))

    def get_newest_children(self):
        """ Returns tuple containing all nodes thaat are the most recent
        children on the tree. This also means due to the structure of
        the tree that they are on the deepest level of the tree.
        If there are no children, the method returns an empty tuple."""
        node_level_list = [node for node in tree.LevelOrderGroupIter(self)]
        most_recent_branch = node_level_list[-1]
        if most_recent_branch[0].is_root:
            return ()
        return most_recent_branch


class DepOptimizer():
    """
    Finds dependencies on a dataset using multi-label
    classification.

    Argument Keywords:
    data -- data object from constants
    f1_threshold -- float in [0, 1]. Potential dependencies need at least this
    f1 score to be recognized
    """

    def __init__(self, data, f1_threshold=0.8):
        """
        Keyword Arguments:
        data : Dataset object from constants.py
            the dataset for which dependencies will be searched
        """
        self.data = data
        self.top_down_convergence = False
        self.f1_threshold = f1_threshold

    def load_data(self):
        """ Loads train/validate/test splits. Sets class-attributes
        df_train, df_validate, df_test and df_columns. """
        df_train, df_validate, df_test = fd.load_dataframes(
            self.data.splits_path,
            self.data.title,
            self.data.missing_value_token)

        no_dupl = fd.check_split_for_duplicates(
            [df_train, df_validate, df_test])

        if no_dupl == 0:
            self.df_train = df_train
            self.df_validate = df_validate
            self.df_test = df_test
            self.columns = list(df_test.columns)
        else:
            e = '''Found duplicates in train/validate/test splits.
            Remove duplicates and run again.'''
            raise ValueError(e)

    def init_roots(self):
        """ Initializes the potential RHS's as roots of the search-
        space tree. """
        self.roots = {}
        # create level 0
        for col in self.columns:
            is_cont = False
            if col in self.data.continuous:
                is_cont = True
            if is_cont:  # threshold is a MSE
                d = pd.concat([self.df_train, self.df_validate, self.df_test])
                thresh = d.loc[:, col].mean()*0.2  # this is bad
            if not is_cont:
                thresh = self.f1_threshold
            self.roots[col] = RootNode(name=col,
                                       is_continuous=is_cont,
                                       threshold=thresh)

    def print_trees(self):
        """ Prints tree of each root. """
        for root in self.roots.values():
            for pre, _, node in tree.RenderTree(root):
                treestr = u"%s%s" % (pre, node.name)
                print(treestr.ljust(8), node.score)

    def run_top_down(self, dry_run=False):
        """ Runs Top-Down approach. """
        self.top_down_convergence = False
        self.load_data()
        self.init_roots()
        steps = 0
        score_fun = self.run_ml_imputer
        if dry_run:
            score_fun = self.generate_scores
        while not self.top_down_convergence:
            steps += 1
            self.get_top_down_candidates()
            score_fun()
            print('step {}'.format(steps))
        print('Found minimal dependencies in {0} steps.'.format(steps))
        self.print_trees()

    def get_top_down_candidates(self):
        """ Generates dependency-candidates based on
        previous results. The top-down approach starts with a
        maximal lhs at level 1. With each additional level,
        columns are dropped until a minimal lhs is reached. """
        # create level 1
        if list(self.roots.values())[0].children == ():
            for root in self.roots.values():
                tree.Node([c for c in self.columns if c != root.name],
                          parent=root, score=None)

        # create level N
        else:
            self.top_down_convergence = True
            for root in self.roots.values():
                most_recent_nodes = root.get_newest_children()
                for node in most_recent_nodes:
                    if root.is_continuous:
                        if (node.score < root.threshold) and (len(node.name) > 1):
                            self.top_down_convergence = False
                            pot_lhs = node.name
                            for col in pot_lhs:
                                tree.Node(
                                    [c for c in pot_lhs if c != col],
                                    parent=node, score=None)

                    elif not root.is_continuous:
                        if (node.score > root.threshold) and (len(node.name) > 1):
                            self.top_down_convergence = False
                            pot_lhs = node.name
                            for col in pot_lhs:
                                tree.Node(
                                    [c for c in pot_lhs if c != col],
                                    parent=node, score=None)

    def run_ml_imputer(self):
        """ Runs Datawig imputer on all nodes on the deepest levels on all
        trees. The root node's name contains the potential rhs to be imputed,
        the node's name the potential lhs."""
        for root in self.roots.values():
            most_recent_nodes = root.get_newest_children()
            for node in most_recent_nodes:
                if node.score is None:
                    print(str(node.name) + ' for RHS ' + str(root.name))
                    dependency = {root.name: [node.name]}
                    res = fd.run_ml_imputer_on_fd_set(self.df_train,
                                                      self.df_validate,
                                                      self.df_test,
                                                      dependency,
                                                      self.data.continuous)
                    if root.is_continuous:
                        node.score = res[root.name][0]['mse']
                    else:
                        node.score = res[root.name][0]['f1']

    def generate_scores(self):
        """ Randomly generates scores for nodes on the deepest levels on all
        trees. """
        for root in self.roots.values():
            most_recent_nodes = root.get_newest_children()
            if not root.is_continuous:
                for node in most_recent_nodes:
                    if node.score is None:
                        node.score = random.random()
            elif root.is_continuous:
                for node in most_recent_nodes:
                    if node.score is None:
                        rng = np.random.normal(root.threshold*1.2,
                                               root.threshold*0.25)
                        node.score = rng
