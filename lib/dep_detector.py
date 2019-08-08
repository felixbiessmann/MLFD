import lib.fd_imputer as fd
import anytree as tree
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
    """

    def __init__(self, name, is_continuous: bool):
        self.name = name
        self.score = ''
        self.is_continuous = is_continuous

    def __str__(self):
        return(str(self.name))

    def get_newest_children(self):
        node_level_list = [node for node in tree.LevelOrderGroupIter(self)]
        most_recent_node = node_level_list[-1]
        return most_recent_node


class DepOptimizer():
    """
    Finds dependencies on a dataset using multi-label
    classification.
    """

    def __init__(self, data, method='top_down'):
        """
        Keyword Arguments:
        data : Dataset object from constants.py
            the dataset for which dependencies will be searched
        """
        self.data = data
        self.method = method
        self.top_down_convergence = False

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
            self.roots[col] = RootNode(name=col, is_continuous=is_cont)

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
        while not self.top_down_convergence:
            steps += 1
            self.get_top_down_candidates()
            self.generate_scores()
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
                    if (node.score > 0.8) and (len(node.name) > 1):
                        self.top_down_convergence = False
                        pot_lhs = node.name
                        for col in pot_lhs:
                            tree.Node([c for c in pot_lhs if c != col],
                                      parent=node, score=None)

    def run_ml_imputer(self):
        """ Runs Datawig imputer on all nodes on the deepest levels on all
        trees. The root node's name contains the potential rhs to be imputed,
        the node's name the potential lhs."""
        for root in self.roots.values():
            most_recent_nodes = root.get_newest_children()
            for node in most_recent_nodes:
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
            for node in most_recent_nodes:
                if node.score is None:
                    node.score = random.random()


def bottom_up_candidates(columns, candidates=[], save=True):
    """ Returns dict of FD candidates with rhs as key and
    a list of lists with list of lhs-columns and f1-score
    as entries.

    Keywords:
    columns -- list of columns to detect dependencies on
    save -- boolean, saves results """
    # No level of results available, create level 0
    if candidates == []:
        candidates.append({})
        for pot_rhs in columns:
            candidates[0][pot_rhs] = []
            for pot_lhs in columns:
                if pot_lhs != pot_rhs:
                    candidates[0][pot_rhs].append([[pot_lhs], ''])

    # there is already one level of results available
    elif len(candidates) == 1:
        pass

    # there are already results in candiates
    elif len(candidates) > 1:
        for pot_rhs in candidates[-1]:
            for result_pair in candidates[-1][pot_rhs]:
                pass

    return candidates


def top_down_candidates(columns, candidates=[], save=True):
    """ Returns dict of FD candidates with rhs as key and a list
    of lists with list of lhs-columns and f1-score as entries.

    Starts with big lhs combination and shrinks them down to minimal
    ones.

    Keywords:
    columns -- list of columns to detect dependencies on
    candidates -- list containing generations of dictionaries with FD
    candidates as keys and list of f1-score and lhs-candidates as columns
    save -- boolean, save the result as pickled dict """


def dep_detector(df_train, df_test, df_validate, candidates):
    """ Returns performance of potential FDs from dict candidates on
    a dataset.

    Keywords:
    data -- data-object
    candidates -- dict with potential rhs as key and list
    of pot_lhs as values """
    df_train, df_validate, df_test = fd.load_dataframes(
        data.splits_path,
        data.title,
        data.missing_value_token)
    fd.check_split_for_duplicates([df_train, df_validate, df_test])
    continuous_cols = data.continuous

    for pot_rhs, list_pot_lhs in candidates.items():
        for key, pot_lhs in enumerate(list_pot_lhs):
            relevant_cols = pot_lhs[0] + [pot_rhs]
            print(relevant_cols)

            if pot_rhs not in continuous_cols:
                train = df_train.iloc[:, relevant_cols].astype(
                    {pot_rhs: str})
                validate = df_validate.iloc[:, relevant_cols].astype(
                    {pot_rhs: str})
                test = df_test.iloc[:, relevant_cols].astype(
                    {pot_rhs: str})
            else:
                train = df_train.iloc[:, relevant_cols]
                validate = df_validate.iloc[:, relevant_cols]
                test = df_test.iloc[:, relevant_cols]

            df_imputed = fd.ml_imputer(train,
                                       validate,
                                       test,
                                       pot_rhs)

            result = fd.get_performance(df_imputed,
                                        pot_rhs,
                                        pot_lhs[0],
                                        continuous_cols)
            if 'f1' in result.keys():
                candidates[pot_rhs][key][1] = result['f1']
            else:
                candidates[pot_rhs][key][1] = result['mse']

    print(candidates)
