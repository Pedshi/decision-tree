from typing import Tuple, Union
import pandas as pd


def gt(a, b) -> bool:
    return a > b


def lte(a, b) -> bool:
    return a <= b


def eq(a, b) -> bool:
    return a == b


def neq(a, b) -> bool:
    return a != b


def gini_impurity(ser) -> float:
    """Return gini impurity for series"""
    pb = ser.value_counts() / ser.shape[0]
    gini = 1 - (pb**2).sum()
    return round(gini, 4)


def ljoin_filter(target, attribute, val, exp) -> pd.Series:
    """Left join target with attribute where condition satisfies exp(val)"""
    mask = exp(attribute, val)
    target_split = target.loc[mask]
    return target_split


def gini_index(target, attribute, val, l_exp, r_exp) -> float:
    """Return gini index for attribute

    Keyword arguments:
    target -- the target
    attribute -- the attribute to split on
    val -- the value to split attribute on
    l_exp -- expression used to create left split
    r_exp -- expression used to create right split
    """

    count_tot = target.shape[0]

    target_l = ljoin_filter(target, attribute, val, l_exp)
    count_l = target_l.shape[0]

    target_r = ljoin_filter(target, attribute, val, r_exp)
    count_r = target_r.shape[0]

    tot = (count_l/count_tot) * gini_impurity(target_l)
    tot += (count_r/count_tot) * gini_impurity(target_r)
    return round(tot, 4)



