from typing import Tuple
import numpy as np
import pandas as pd
from typing import Union


def gte(a, b) -> bool:
    return a >= b


def lt(a, b) -> bool:
    return a < b


def eq(a, b) -> bool:
    return a == b


def entropy(ser) -> float:
    """Return entropy for series"""
    pb = ser.value_counts() / ser.shape[0]
    e = (-pb * np.log2(pb)).sum()
    return round(e, 4)


def gini_impurity(ser) -> float:
    """Return gini impurity for series"""
    pb = ser.value_counts() / ser.shape[0]
    gini = (pb * (1-pb)).sum()
    return round(gini, 4)


def ljoin_filter(target, attribute, split_val, exp) -> pd.Series:
    """Left join target and attribute with condition exp(split_val)"""
    mask = exp(attribute, split_val)
    target_split = target.loc[mask]
    return target_split


def ljoin_filter_count(target, attribute, split_val, exp) -> Tuple[pd.Series, int]:
    """Return left join target and attribute with condition exp(split_val), and number of rows"""
    target_split = ljoin_filter(target, attribute, split_val, exp)
    count = target_split.shape[0]
    return target_split, count


def information_gain_cat(target, attribute) -> Tuple[float, np.array]:
    """Return highest information gain and values"""

    e_target = entropy(target)
    n = attribute.shape[0]
    values = attribute.unique()
    tot = 0

    for val in values:
        target_after = target.loc[attribute == val]
        m = target_after.shape[0]
        tot += (m/n) * entropy(target_after)

    ig = e_target - tot
    ig = round(ig, 4)
    return ig, values


def information_gain_split(e_target, target, attribute, split_val, n) -> float:
    """
    Return information gain for target split in lower and
    larger split of split_val
    """
    (target_lt, m_lt) = ljoin_filter_count(target, attribute, split_val, lt)
    (target_gte, m_gte) = ljoin_filter_count(target, attribute, split_val, gte)
    tot = (m_lt/n) * entropy(target_lt)
    tot += (m_gte/n) * entropy(target_gte)
    return e_target - tot


def information_gain_num(target, attribute) -> Tuple[float, Union[int, list]]:
    """Return highest information gain and value among all unique values in attribute"""

    e_target = entropy(target)
    n = attribute.shape[0]
    values = attribute.unique()
    best_ig = 0
    for val in values:
        curr_split_ig = information_gain_split(e_target, target, attribute, val, n)
        if curr_split_ig > best_ig:
            best_ig = round(curr_split_ig, 4)
            best_val = val

    return best_ig, best_val


def information_gain(target, attribute) -> Tuple[float, Union[int, list], bool]:
    """Return information gain for attribute, best value/values and if attribute is numeric.

    If attribute type is numeric then the value giving the best information gain is only used to calculate
    information gain, for other types all possible values are used to calculate information gain
    """
    if attribute.dtype == 'O':
        (ig, val) = information_gain_cat(target, attribute)
        is_numeric = False
    else:
        (ig, val) = information_gain_num(target, attribute)
        is_numeric = True
    return ig, val, is_numeric
