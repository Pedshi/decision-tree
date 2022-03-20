from typing import Tuple, Union, TypedDict, Callable, List

import pandas as pd
from tree.node import Node
from tree.branch import Branch
from tree.leaf import Leaf
from cart.cart_utils import *


class SplitWithInfo(TypedDict):
    data: pd.Series
    exp: Callable
    val: Union[float, str]


class CARTTree:

    def __init__(self, max_depth: int = None):
        self.max_depth = max_depth
        self.root = None

    def train(self, data: pd.DataFrame, target: pd.Series):
        self.root = self._train_tree(data, target, 0)

    def _train_tree(self, data: pd.DataFrame, target: pd.Series, depth: int):
        """Recursively create subtrees that maximizes purity of target """

        if is_empty(target):
            return None
        if self.should_predict(data, target, depth):
            pred = make_prediction(target)
            return Leaf(pred)

        (ig, attr_name, vals, is_numeric) = self._best_split(data, target)

        best_attr = data.pop(attr_name)

        target_splits = self._make_split(target, best_attr, vals, is_numeric)

        node = Node(ig, attr_name)

        for split in target_splits:
            branch = Branch(split['val'], split['exp'])
            child = self._train_tree(data, split['data'], depth - 1)

            if child is None:
                continue

            branch.child = child
            node.add_branch(branch)

        return node

    def should_predict(self, data: pd.DataFrame, target: pd.Series, depth: int) \
            -> bool:
        """Return true if any of stop criterias are reached else false"""
        return depth == self.max_depth or \
               contains_one_type(target) or is_empty(data)

    def _best_split(self, data: pd.DataFrame, target: pd.Series) \
            -> Tuple[float, str, Union[float, str], bool]:
        """Return the value, attribute name and purity of the attribute with lowest impurity

        Finds the val with lowest impurity for each atrribute and compares
        to pick the one with lowest impurity.
        """
        least_impure = 1
        best_val = None
        best_is_numeric = None
        best_arg_name = None

        columns = data.columns

        for col in columns:
            attribute = data[col]
            (impure, val, is_numeric) = self._best_split_value(target, attribute)
            if impure < least_impure:
                least_impure = impure
                best_val = val
                best_is_numeric = is_numeric
                best_arg_name = col

        return least_impure, best_arg_name, best_val, best_is_numeric

    def _make_split(self, target: pd.Series, attribute: pd.Series,
                    val: Union[float, str], is_numeric: bool) \
            -> List[SplitWithInfo]:
        """Split target into two on val"""

        data_splits = []

        if is_numeric:
            l_exp = lte
            r_exp = gt
        else:
            l_exp = eq
            r_exp = neq

        target_l = ljoin_filter(target, attribute, val, l_exp)
        target_r = ljoin_filter(target, attribute, val, r_exp)
        data_splits.append({'data': target_l, 'exp': l_exp, 'val': val})
        data_splits.append({'data': target_r, 'exp': r_exp, 'val': val})

        return data_splits

    def _best_split_value(self, target: pd.Series, attribute: pd.Series) \
            -> Tuple[float, Union[float, str], bool]:
        """Return value with the lowest gini_index"""

        is_numeric = attribute.dtype == 'O'

        if is_numeric:
            l_exp = lte
            r_exp = gt
        else:
            l_exp = eq
            r_exp = neq

        best_gini = 1
        best_val = None
        for val in attribute:
            gini = gini_index(target, attribute, val, l_exp, r_exp)
            if gini < best_gini:
                best_gini = gini
                best_val = val

        return gini, best_val, is_numeric

    def predict(self, data: pd.DataFrame) -> Union[float, str]:
        """Predict target for data"""
        return self._predict(data, self.root)

    def _predict(self, data: pd.DataFrame, node: Node) -> Union[float, str]:
        """Recursively follows conditions in branchs to find
        prediction, starting at node"""

        if isinstance(node, Leaf):
            return node.prediction

        val = data[node.attr_name].iloc[0]

        for branch in node.branchs:
            if branch.exp(val, branch.val):
                return self._predict(data, branch.child)

        return None

    def print_tree(self) -> None:
        self._print_tree(self.root)

    def _print_tree(self, node: Node) -> None:
        if isinstance(node, Leaf):
            print(f'Pred: {node.prediction}')
            return
        for branch in node.branchs:
            print(f'{node.attr_name} -- {branch.val} {branch.exp.__name__} --> ', end='')
            self._print_tree(branch.child)
            print('------')


def make_prediction(target):
    if target.dtype == 'O':
        return predict_categorical(target)
    else:
        return predict_regression(target)


def predict_categorical(target):
    target_freq = target.value_counts().sort_values(ascending=False)
    return target_freq.index[0]


def predict_regression(target):
    return target.mean()


def contains_one_type(target):
    return target.unique().shape[0] == 1


def is_empty(data):
    return data.empty