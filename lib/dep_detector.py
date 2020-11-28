import lib.fd_imputer as fd
import anytree as tree
import pandas as pd
import numpy as np
import random


def get_continuous_min_dep(root):
    """ Finds minimal LHS combinations on a tree of a continuous
    RHS. All node.name on the tree need to be lists for this function
    to work properly, excluding the root.

    Keyword Arguments:
    root -- root of a tree where minimal LHS combinations are to be
    searched. all children's names need to be a list.
    strategy -- 'greedy' or 'complete'. Strategy with which tree has been
    created. """
    candidates = {}
    if root.search_strategy == 'complete':
        for node in tree.LevelOrderIter(root.children[0]):
            if node.score <= node.parent.score*0.98:
                candidates[tuple(node.name)] = node.score
                parent = node.parent
                if not parent.is_root:
                    try:
                        del candidates[tuple(node.parent.name)]
                    except KeyError:
                        pass
    elif root.search_strategy == 'greedy':
        parent_is_minimal = True
        newest_children = root.get_newest_children()
        for child in newest_children:
            if child.score <= 0.98*child.parent.score:
                candidates[tuple(child.name)] = child.score
                parent_is_minimal = False
        if parent_is_minimal:
            parent = newest_children[0].parent
            candidates[tuple(parent.name)] = parent.score
    return candidates


def get_non_continuous_min_dep(root):
    """ Finds minimal LHS combinations on a tree of a non-continuous
    RHS. All node.name on the tree need to be lists for this function
    to work properly, excluding the root.

    Keyword Arguments:
    root -- root of a tree where minimal LHS combinations are to be
    searched. all children's names need to be a list.
    strategy -- 'greedy' or 'complete'. Strategy with which tree has been
    created. """
    candidates = {}
    if root.search_strategy == 'complete':
        for node in tree.LevelOrderIter(root.children[0]):
            if node.score >= node.parent.score*0.98:
                candidates[tuple(node.name)] = node.score
                parent = node.parent
                if not parent.is_root:
                    try:
                        del candidates[tuple(node.parent.name)]
                    except KeyError:
                        pass
    elif root.search_strategy == 'greedy':
        parent_is_minimal = True
        newest_children = root.get_newest_children()
        for child in newest_children:
            if child.score >= 0.98*child.parent.score:
                candidates[tuple(child.name)] = child.score
                parent_is_minimal = False
        parent = newest_children[0].parent
        if parent_is_minimal and (not parent.is_root):
            candidates[tuple(parent.name)] = parent.score
    return candidates


class RootNode(tree.NodeMixin):
    """
    Root node class for dependency search-trees.

    Methods:
    get_newest_children -- Returns node(s) that is/are the most recent child
        or children of a tree

    Attributes:
    name -- potential rhs column name
    score -- threshold that child-nodes are compared to when searching for
    dependencies. Either MSE or f1-score, depending on is_continuous.
    is_continuous -- boolean, True if column contains continuous values, False
    if values are classifiable
    search_strategy -- string, either "greedy" or "complete". Determines the
    search-strategy used for detecting dependencies. Use "complete" to find all
    dependencies on the tree and "greedy" to find one high-performing
    dependency.
    known_scores -- dict, stores all ml_imputer() results for particular LHS
    combinations. Strongly improves performance when searching for depdencies.
    training_cycles -- int, number of training cycles run by Datawig when
    training models. Lower number worsens model performance but reduces
    training time strongly.
    """

    def __init__(self, name, columns, train, validate, test,
                 continuous, is_continuous: bool, threshold=None,
                 training_cycles=10):
        self.name = name
        self.score = threshold
        self.is_continuous = is_continuous
        self.columns = columns
        self.top_down_convergence = False

        self.df_train = train
        self.df_validate = validate
        self.df_test = test
        self.continuous = continuous

        self.search_strategy = ''  # greedy or complete
        self.cycles = training_cycles

        self.known_scores = {}

        # init first child-Node
        tree.Node([c for c in self.columns if c != self.name],
                  parent=self, score=None)

    def __str__(self):
        return(str(self.name))

    def print_tree(self):
        """ Prints tree of the root. """
        for pre, _, node in tree.RenderTree(self):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.score)

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

    def run_top_down(self, strategy='greedy', dry_run=False):
        """ Runs Top-Down approach. Generates scores, if dry_run is
        True. Uses either a greedy strategy or a complete strategy
        to find dependencies."""
        self.search_strategy = strategy
        score_fun = self.run_ml_imputer
        if dry_run:
            score_fun = self.generate_scores
        if strategy == 'greedy':
            candidates_generator = self.get_greedy_candidates
        elif strategy == 'complete':
            candidates_generator = self.get_complete_candidates
        else:
            raise ValueError('''Indicate a valid strategy for
                    dependency detection - either greeedy or
                    complete''')
        steps = 0
        self.top_down_convergence = False
        while not self.top_down_convergence:
            steps += 1
            score_fun()
            candidates_generator()
            print('step {}'.format(steps))
        print('Found minimal dependencies in {0} steps.'.format(steps))
        self.print_tree()

    def get_greedy_candidates(self):
        """ Generates depdendency-candidates based on a greedy strategy.
        This strategy tries to find _one_ minimal lhs that performs
        strongly to save computation time.
        Neither is the lhs found a minimal lhs, nor are all relevant lhs
        detected.
        """
        self.top_down_convergence = True
        most_recent_nodes = self.get_newest_children()
        highscore = most_recent_nodes[0].parent.score
        highscore_node = None
        for node in most_recent_nodes:
            if self.is_continuous:
                if (node.score <= 1.02*highscore) and (len(node.name) > 1):
                    highscore_node = node

            elif not self.is_continuous:
                if (node.score >= 0.98*highscore) and (len(node.name) > 1):
                    highscore_node = node

        if highscore_node is not None:
            self.top_down_convergence = False
            pot_lhs = highscore_node.name
            for col in pot_lhs:
                tree.Node(
                    [c for c in pot_lhs if c != col],
                    parent=highscore_node, score=None)

    def get_complete_candidates(self):
        """ Generates dependency-candidates based on a strategy trying to
        find all dependencie. The top-down approach starts with a
        maximal lhs at level 1. With each additional level,
        columns are dropped until a minimal lhs is reached. """
        self.top_down_convergence = True
        most_recent_nodes = self.get_newest_children()
        for node in most_recent_nodes:
            if self.is_continuous:
                if (node.score < node.parent.score) and (len(node.name) > 1):
                    self.top_down_convergence = False
                    pot_lhs = node.name
                    for col in pot_lhs:
                        tree.Node(
                            [c for c in pot_lhs if c != col],
                            parent=node, score=None)

            elif not self.is_continuous:
                if (node.score > node.parent.score*0.98) \
                        and (len(node.name) > 1):
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
        most_recent_nodes = self.get_newest_children()
        for node in most_recent_nodes:
            print(str(node.name) + ' for RHS ' + str(self.name))
            # search for known scores
            node.score = self.known_scores.get(tuple(node.name), None)
            if node.score is None:
                dependency = {self.name: [node.name]}
                res = fd.run_ml_imputer_on_fd_set(self.df_train,
                                                  self.df_validate,
                                                  self.df_test,
                                                  dependency,
                                                  self.continuous,
                                                  self.cycles)
                if self.is_continuous:
                    node.score = res[self.name][0]['mse']
                else:
                    node.score = res[self.name][0]['f1']

                # add score to dict
                self.known_scores[tuple(node.name)] = node.score

    def generate_scores(self):
        """ Randomly generates scores for nodes on the deepest levels on all
        trees. """
        most_recent_nodes = self.get_newest_children()
        if not self.is_continuous:
            for node in most_recent_nodes:
                if node.score is None:
                    node.score = random.random()
        elif self.is_continuous:
            for node in most_recent_nodes:
                if node.score is None:
                    rng = np.random.normal(node.parent.score*1.2,
                                           node.parent.score*0.25)
                    node.score = rng

    def extract_minimal_deps(self):
        """
        Finds all minimal LHS combinations and returns them in a dict.
        """
        if not self.is_continuous:
            measure = 'F1-Score'
            min_lhs = get_non_continuous_min_dep(self)
        else:
            measure = 'Mean Squared Error'
            min_lhs = get_continuous_min_dep(self)
        print(f'\nLHS combinations for RHS {self.name}:')
        if min_lhs == {}:
            print('No minimal dependencies were detected.')
        else:
            for lhs in min_lhs:
                print('{} with {} {:5.4f}'.format(list(lhs),
                                                  measure,
                                                  min_lhs[lhs]))
        return min_lhs


class DepOptimizer():
    """
    Finds dependencies on a dataset using multi-label
    classification.

    Argument Keywords:
    data -- data object from constants
    f1_threshold -- float in [0, 1]. Potential dependencies need at least this
    f1 score to be recognized
    """

    def __init__(self, data, f1_threshold=0.9):
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
                thresh = d.loc[:, col].mean()*0.4  # this is bad
            if not is_cont:
                thresh = self.f1_threshold
            self.roots[col] = RootNode(name=col,
                                       train=self.df_train,
                                       validate=self.df_validate,
                                       test=self.df_test,
                                       continuous=self.data.continuous,
                                       columns=self.columns,
                                       is_continuous=is_cont,
                                       threshold=thresh)

    def print_trees(self):
        """ Prints tree of each root. """
        for root in self.roots.values():
            root.print_tree()
        return True

    def search_dependencies(self, strategy='greedy', dry_run=False):
        """ Searches all columns of the original database table for
        dependencies. """
        self.load_data()
        self.init_roots()
        for root in self.roots.values():
            root.run_top_down(strategy, dry_run)
        return True

    def get_minimal_dependencies(self):
        """ Yields and prints the minimal LHS combinations for all
        root nodes"""
        for root in self.roots.values():
            root.extract_minimal_deps()
        return True
