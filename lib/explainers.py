import pandas as pd
import shap
import numpy as np
from autogluon.tabular import TabularPredictor
from typing import List, Tuple


SHAPs = List[np.array]
FlatBenchmarks = List[Tuple[str, float, float]]


def run_feature_permutation(predictor: TabularPredictor,
                            df_train: pd.DataFrame,
                            model_name=None) -> pd.DataFrame:
    """
    Use feature permutation to derive feature importances from an AutoGluon
    model. The AG documentation refers this website to explain feature
    permutation: https://explained.ai/rf-importance/
    """
    df_importance = predictor.feature_importance(df_train,
                                                 model=model_name,
                                                 num_shuffle_sets=10,
                                                 subsample_size=1000)
    return df_importance


def get_n_best_features(n: int, global_shaps: np.array) -> List[int]:
    """
    Given an array of floats, calculates the absolute value of each component
    and returns the indices of those components as a sorted list, with the
    indice of the biggest absolute component first and the smallest absolute
    component last.
    """
    cols = [(i_col, shap) for i_col, shap in enumerate(global_shaps)]

    # sort columns by absolute SHAP values
    sorted_cols = sorted(cols, key=lambda t: abs(t[1]), reverse=True)
    list_of_indices = [t[0] for i, t in enumerate(sorted_cols) if i < n]
    return list_of_indices


# SHAP functions
class AGWrapper:
    """
    A wrapper around AutoGluon that makes it callable by shap.KernelExplainer.
    The APIs of AG and SHAP are not compatible otherwise unfortunately

    The code is copied from one of the main contributors of AG, the original
    code is here:
    https://github.com/awslabs/autogluon/tree/master/examples/tabular/interpret
    """

    def __init__(self, predictor, feature_names, model_name=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.model_name = model_name

    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, model=self.model_name)


def train_shap_explainer(predictor: TabularPredictor, X_train: pd.DataFrame,
                         model_name=None):
    baseline = X_train.sample(250)  # TODO read up what this does exactly
    ag_wrapper = AGWrapper(predictor, X_train.columns, model_name=model_name)
    explainer = shap.KernelExplainer(model=ag_wrapper.predict_proba,
                                     data=baseline)
    return explainer


def calculate_shap_values(explainer: shap.explainers, X_test: pd.DataFrame,
                          nsamples: int = 50):
    shap_values = explainer.shap_values(X=X_test, nsamples=nsamples)
    return shap_values


def calculate_global_shap_values(shap_values: SHAPs,
                                 problem_type: str,
                                 df_label_true: pd.DataFrame) -> np.array:
    """
    The shap_values come in form of a list of matrices. That list has the same
    length as the classification problem has classes. For each matrix in that
    list, calculate the average of each column. You then get a list of lists,
    which you can interpret as a matrix with as many columns as the original
    table has attributes and as many rows as the classification problem has
    classes. Calculate the average weighted by occurance of the classes in the
    dataset, and you get global feature importances.
    """
    if problem_type == 'binary':
        class_shaps = [np.mean(s, axis=0) for s in shap_values]
        counts = df_label_true.value_counts().values
        weights = counts/df_label_true.shape[0]
        return np.average(np.array(class_shaps), axis=0, weights=weights)
    else:
        raise ValueError(f"""Global SHAP computation for AG problem type
            {problem_type} hasn't been implemented yet!""")
