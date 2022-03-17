from typing import Tuple


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


def gini_index(target, attribute, val, n, l_exp, r_exp) -> float:
    """Return gini index for attribute

    Keyword arguments:
    target -- the target
    attribute -- the attribute to split on
    val -- the value to split attribute on
    n -- total number of instances in target
    l_exp -- expression used to create left split
    r_exp -- expression used to create right split
    """

    (target_lt, m_lt) = ljoin_filter_count(target, attribute, val, l_exp)
    (target_gte, m_gte) = ljoin_filter_count(target, attribute, val, r_exp)
    tot = (m_lt / n) * gini_impurity(target_lt)
    tot += (m_gte / n) * gini_impurity(target_gte)
    return round(1 - tot, 4)

