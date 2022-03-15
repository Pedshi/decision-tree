from tokenize import Double
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
  pb = ser.value_counts() / ser.shape[0]
  E = (-pb * np.log2(pb)).sum()
  return round(E, 4)

def gini_impurity(ser) -> float:
  pb = ser.value_counts() / ser.shape[0]
  G = (pb * (1-pb)).sum()
  return round(G, 4)

def ljoin_filter(target, attribute, split_val, exp) -> pd.Series:
  mask = exp(attribute, split_val)
  target_split = target.loc[mask]
  return target_split

def ljoin_filter_count(target, attribute, split_val, exp) -> Tuple[pd.Series, int]:
  target_split = ljoin_filter(target, attribute, split_val, exp)
  count = target_split.shape[0]
  return (target_split, count)

def information_gain_cat(target, attribute) -> Tuple[float, np.array]:
  E_target = entropy(target)
  n = attribute.shape[0]
  values = attribute.unique()
  sum = 0
  
  for val in values:
    target_after = target.loc[attribute == val]
    m = target_after.shape[0]
    sum += (m/n) * entropy(target_after)
  
  ig = E_target - sum
  ig = round(ig, 4)
  return (ig, values)

def information_gain_split(E_target, target, attribute, split_val, n) -> float:
  (target_lt, m_lt) = ljoin_filter_count(target, attribute, split_val, lt)
  (target_gte, m_gte) = ljoin_filter_count(target, attribute, split_val, gte)
  sum = (m_lt/n) * entropy(target_lt)
  sum += (m_gte/n) * entropy(target_gte)
  return E_target - sum

def information_gain_num(target, attribute) -> Tuple[float, Union[int, list]]:
  E_target = entropy(target)
  n = attribute.shape[0]
  values = attribute.unique()
  best_ig = 0
  for val in values:
    curr_split_ig = information_gain_split(E_target, target, attribute, val, n)
    if curr_split_ig > best_ig:
      best_ig = round(curr_split_ig, 4)
      best_val = val

  return (best_ig, best_val)

def information_gain(target, attribute) -> Tuple[float, Union[int, list], bool]:
  if attribute.dtype == 'O':
    (ig, val) = information_gain_cat(target, attribute)
    is_numeric = False
  else :
    (ig, val) = information_gain_num(target, attribute)
    is_numeric = True
  return (ig, val, is_numeric)