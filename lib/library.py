from autogluon.tabular import TabularPredictor as task
from sklearn.model_selection import train_test_split
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
    name -- rhs-candidate column name
    score -- threshold that child-nodes are compared to when searching for
    dependencies. Either MSE or f1-score, depending on is_continuous.
    is_continuous -- boolean, True if column contains continuous values, False
    if values are classifiable
    search_strategy -- string, either "greedy" or "complete". Determines the
    search-strategy used for detecting dependencies. Use "complete" to find all
    dependencies on the tree and "greedy" to find one high-performing
    dependency.
    known_scores -- dict, stores all ml_imputer() results for particular LHS
    combinations. Greatly improves performance when searching for depdencies.
    training models. Lower number worsens model performance but reduces
    training time strongly.
    """

    def __init__(self, name, columns, train, validate, test,
                 continuous, is_continuous: bool, threshold=None):
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
                res = run_ml_imputer_on_fd_set(self.df_train,
                                                  self.df_validate,
                                                  self.df_test,
                                                  dependency,
                                                  self.continuous)
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
        df_train, df_validate, df_test = load_dataframes(
            self.data.splits_path,
            self.data.title,
            self.data.missing_value_token)

        no_dupl = check_split_for_duplicates(
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


def split_df(data_title, df, split_ratio, splits_path=''):
    """ Splits a dataframe into train-, validate - and test-subsets.
    If a splits_path is provided, splits will be saved to the harddrive.

    Returns a tuple(df_train, df_validate, df_test) if the splits_path
    is an empty string.

    Keyword arguments:
    data_title - - a string naming the data
    df - - a pandas data frame
    splits_path - - file-path to folder where splits should be saved
    split_ratio - - train/test/validate set ratio, e.g. [0.8, 0.1, 0.1]
    """
    import os
    import numpy as np

    train_ratio, validate_ratio, test_ratio = (split_ratio)

    # compare first col with index. if not equal...
    if not np.array_equal(df.iloc[:, 0].values, np.array(df.index)):
        # ...set index as 0th column so that fd_imputer can use it to impute
        print('No double index detected.')
        df = index_as_first_column(df)

    train_path = f'{splits_path}train/'
    validate_path = f'{splits_path}validate/'
    test_path = f'{splits_path}test/'

    for p in [train_path, validate_path, test_path]:
        if (not os.path.exists(p) and splits_path != ''):
            os.mkdir(p)

    rest_df, test_df = train_test_split(df, test_size=test_ratio)
    train_df, validate_df = train_test_split(rest_df, test_size=validate_ratio)

    if splits_path == '':
        return(train_df, validate_df, test_df)

    else:
        try:
            df.to_csv(splits_path+data_title+'.csv', header=None, sep=',',
                      index=False)
            print(f'Dataset successfully written to \
                    {splits_path}{data_title}.csv')
        except TypeError:
            print(f'Could not save dataframe to {splits_path}{data_title}')

        for name, df, path in [('train', train_df, train_path),
                               ('test', test_df, test_path),
                               ('validate', validate_df, validate_path)]:
            try:
                save_path = f'{path}{data_title}_{name}.csv'
                df.to_csv(save_path, sep=',',
                          index=False, header=None)
                print(f'{name} set successfully written to {save_path}.')
            except TypeError:
                print("Something went wrong writing the splits to files.")


def check_split_for_duplicates(list_of_dfs):
    """ Prints and returns number of duplicate entries in a list of dfs """
    no_duplicates = sum(pd.concat(list_of_dfs).duplicated())
    print(str(no_duplicates) + ' duplicates detected in the splittings.')
    return no_duplicates


def load_dataframes(splits_path, data_title, missing_value_token):
    """ Loads train, validate, test splits from a directory.
    The data's missing values, which are represented in the data by
    missing_value_token, are deleted.

    Argument Keywords:
    splits_path -- path where subdirectories with splits are located
    data_title -- name of the dataset and it's associated splits
    missing_value_token -- specifies how missing values are represented
    in the dataset
    """

    import numpy as np
    df_train = pd.read_csv(splits_path+'train/' +
                           data_title+'_train.csv', header=None)
    df_validate = pd.read_csv(
        splits_path+'validate/'+data_title+'_validate.csv', header=None)
    df_test = pd.read_csv(splits_path+'test/' +
                          data_title+'_test.csv', header=None)

    dfs = [df_train, df_validate, df_test]
    dfs = [df.replace(missing_value_token, np.nan) for df in dfs]
    dfs = [df.dropna(axis=0) for df in dfs]  # drop rows with nans

    return (dfs)


def random_dependency_generator(columns, n=10):
    """ Generates n random dependencies from a list of columns.

    Returns a dictionary with rhs's as keys and associated with it lhs
    combinations, representing a total of <= n dependencies.
    Note that the returned lhs-combinations are neither randomly
    distributed, nor are there exactly n dependencies returned.
    The way rand_length is generated makes it more likely for short
    lhs-combinations to be generated over long lhs-combinations.

    Keyword attributes:
    columns -- list of columns-names
    n -- int, indicating how many dependencies shall be returned
    """
    import random
    dependencies_dict = {}

    for i in range(0, n):
        # at least 2 columns are necessary to form a dependency
        rand_length = random.randint(2, len(columns))
        lhs = random.sample(columns, rand_length)
        rhs = lhs.pop()
        lhs.sort()

        if rhs in dependencies_dict:
            if lhs not in dependencies_dict[rhs]:
                dependencies_dict[rhs].append(lhs)
        else:
            dependencies_dict[rhs] = [lhs]

    return dependencies_dict


def read_fds(fd_path):
    """  Returns a dictionary with FDs

    Functionally determined column serve as keys and arrays of functionally
    determining column-combinations as values.

    Keyword arguments:
    fd_path -- file-path to a metanome result of a FD detection algorithm
    """
    import re
    fd_dict = {}
    save_fds = False
    with open(fd_path) as f:
        for line in f:
            if save_fds:
                line = re.sub('\n', '', line)
                splits = line.split("->")

                # Convert to int and substract 1 to
                # start indexing of columns at 0
                lhs = [(int(x)-1) for x in splits[0].split(',')]
                rhs = int(splits[1])-1

                if rhs in fd_dict:
                    fd_dict[rhs].append(lhs)
                else:
                    fd_dict[rhs] = [lhs]

            if line == '# RESULTS\n':  # Start saving FDs
                save_fds = True

    return fd_dict


def get_performance(df_imputed, rhs: str, lhs: list, continuous_cols: list):
    """ Create a dictionary containing metrics to measure the perfor-
    mance of a classifier. If the classified column contains continuous
    values, return a dictionary with keys {'nans', 'lhs', 'mse'}.
    'mse' is the mean squared error. If the classified column contains
    discrete values, return a dictionary with keys {'lhs', 'f1', 'recall',
    'precision'}.
    Keyword arguments:
    df_imputed -- dataframe. Column names are expected to be strings.
        One column needs to be called rhs, another one rhs+'_imputed'.
    rhs -- string, name of the column that has been imputed
    lhs -- list of strings, lhs of the FD
    continuous_cols -- list of strings, names of columns containing
    continuous values
    """
    from sklearn import metrics

    # turn everything into strings to facilitate df selection
    rhs = str(rhs)
    lhs = list(map(str, lhs))
    df_imputed.columns = list(map(str, df_imputed.columns))
    continuous_cols = list(map(str, continuous_cols))

    if rhs in continuous_cols:  # ignore NaNs, count them
        result_selector = df_imputed.loc[:, rhs+'_imputed'].isna()
        y_true = df_imputed.loc[~result_selector, rhs]
        y_pred = df_imputed.loc[~result_selector, rhs+'_imputed']
        no_nans = result_selector.sum()
    else:  # include NaNs, adjust dtype
        if isinstance(df_imputed.loc[0, rhs], str):
            df_imputed = df_imputed.fillna('no value')
        else:
            df_imputed = df_imputed.fillna(123456789)
        y_true = df_imputed.loc[:, rhs]
        y_pred = df_imputed.loc[:, rhs+'_imputed']

    if rhs in continuous_cols:
        mse = ''
        if len(y_pred) > 0:
            mse = metrics.mean_squared_error(y_true, y_pred)

        result = {'nans': no_nans, 'lhs': lhs, 'mse': mse}
    else:
        result = {
            'lhs': lhs,
            'f1': metrics.f1_score(y_true, y_pred, average='weighted'),
            'recall': metrics.recall_score(y_true,
                                           y_pred,
                                           average='weighted'),
            'precision': metrics.precision_score(y_true,
                                                 y_pred,
                                                 average='weighted')
        }
    return result


def ml_imputer(df_train, df_validate, df_test, label_column: str):
    train_data = task.Dataset(df_train)
    test_data = task.Dataset(df_test)
    validate_data = task.Dataset(df_validate)
    train_data.columns = [str(i) for i in df_train.columns]
    test_data.columns = [str(i) for i in df_test.columns]
    validate_data.columns = [str(i) for i in df_validate.columns]

    d = 'agModels-predictClass'  # folder to store trained models
    predictor = task(label=str(label_column),
                     path=d).fit(train_data=train_data, tuning_data=test_data)

    validate_data_no_y = validate_data.drop(labels=[label_column], axis=1)
    y_pred = predictor.predict(validate_data_no_y)
    print("These are the predicted labels: ")
    print(y_pred)
    print("And this is the validation-dataset: ")
    print(validate_data)

    df_imputed = y_pred.to_frame(name=f'{label_column}_imputed')
    print('df_imputed: ')
    print(df_imputed)

    df_validate_imputed = pd.merge(df_validate.loc[:, label_column],
                                   df_imputed,
                                   left_index=True,
                                   right_index=True,
                                   how='left')
    return df_validate_imputed


def run_ml_imputer_on_fd_set(df_train, df_validate, df_test, fds,
                             continuous_cols=[]):
    """ Runs ml_imputer for every fd contained in a dictionary on a split df.

    Executes ml_imputer() for every fd in fds. df_train and df_split should be
    created using split_df(). continuous_cols is used to distinguish between
    columns containing continuous_cols numbers and classifiable objects.

    Keyword arguments:
        df_train -- a df on which fds have been detected.
        df_validate -- a df on which no fds have been detected.
        fds -- a dictionary with an fd's possible rhs as values and a list of
        lhs-combinations as value of fds[rhs].

    Returns:
    A dictionary consisting of rhs's as keys with associated performance
    metrics for each lhs. Performance metrics are precision, recall and
    f1-measure for classifiable data and the mean squared error as well as
    the number of unsucessful imputations for continuous data.
    """
    ml_imputer_results = {}
    for rhs in fds:
        rhs_results = []
        for lhs in fds[rhs]:
            relevant_cols = lhs + [rhs]
            df_subset_train = df_train.iloc[:, relevant_cols]
            df_subset_validate = df_validate.iloc[:, relevant_cols]
            df_subset_test = df_test.iloc[:, relevant_cols]

            ml_result = ml_imputer(df_subset_train,
                                   df_subset_validate,
                                   df_subset_test,
                                   str(rhs))

            result = get_performance(ml_result,
                                     rhs, lhs, continuous_cols)
            rhs_results.append(result)
        ml_imputer_results[rhs] = rhs_results
    return ml_imputer_results


def index_as_first_column(df):
    df = df.reset_index()
    df.columns = [x for x in range(0, len(df.columns))]
    return df
