import numpy as np
from tree.node import Node
from tree.branch import Branch
from tree.leaf import Leaf
from tree.util import *
import pandas as pd

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

def isEmpty(data):
  return data.empty


class DescisionTree:

  def __init__(self, max_depth=None):
    self.max_depth = max_depth
    self.root = None

  def train(self, data, target):
    self.root = self._train_tree(data, target, 0)

  def _train_tree(self, data, target, depth):
    if isEmpty(target):
      return None
    if self.shouldPredict(data, target, depth):
      pred = make_prediction(target)
      return Leaf(pred)
    
    (ig, attr_name, vals, is_numeric) = self._best_split_value(data, target)

    best_attr = data.pop(attr_name)

    target_splits = self._make_split(target, best_attr, vals, is_numeric)

    node = Node(ig, attr_name)

    for split in target_splits:
      branch = Branch(split['val'], split['exp'])
      child = self._train_tree(data, split['data'], depth - 1)

      if child == None:
        continue

      branch.child = child
      node.addBranch(branch)

    return node
  
  def shouldPredict(self, data, target, depth):
    return depth == self.max_depth or \
      contains_one_type(target) or isEmpty(data)

  def _best_split_value(self, data, target):
    best_ig = 0
    best_val = None
    best_is_numeric = None
    best_arg_name = None

    columns = data.columns

    for col in columns:
      attribute = data[col]
      (ig, val, is_numeric) = information_gain(target, attribute)
      if ig > best_ig:
        best_ig = ig
        best_val = val
        best_is_numeric = is_numeric
        best_arg_name = col

    return (best_ig, best_arg_name, best_val, best_is_numeric)
  
  def _make_split(self, target, attribute, val, is_numeric):
    '''
      val: int representing best split value for numeric attribute IF is_numeric = true
      val: list representing all values in categorical attribute IF is_numeric = false
    '''
    data_splits = []

    if is_numeric:
      target_lt = ljoin_filter(target, attribute, val, lt)
      target_gte = ljoin_filter(target, attribute, val, gte)
      data_splits.append({'data': target_lt, 'exp': lt, 'val': val})
      data_splits.append({'data': target_gte, 'exp': gte, 'val': val})

    else:
      for cat in val:
        target_split = ljoin_filter(target, attribute, cat, eq)
        data_splits.append({'data': target_split, 'exp': eq, 'val': cat})

    return data_splits
  
  def predict(self, data):
    return self._predict(data, self.root)

  def _predict(self, data, node):
    if isinstance(node, Leaf):
      return node.prediction

    val = data[node.attr_name].iloc[0]

    for branch in node.branchs:
      if branch.exp(val, branch.val):
        return self._predict(data, branch.child)

    return None

  def print_tree(self):
    self._print_tree(self.root)

  def _print_tree(self, node):
    if (isinstance(node, Leaf)):
      print(f'Pred: {node.prediction}')
      return
    for branch in node.branchs:
      print(f'{node.attr_name} -- {branch.val} {branch.exp.__name__} --> ', end='')
      self.print_tree(branch.child)
      print('------')