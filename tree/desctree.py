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

  def train(self, target, training):
    self.root = self._train_tree(target, training, 0)

  def _train_tree(self, target, training, depth):
    if isEmpty(target):
      return None

    if depth == self.max_depth or contains_one_type(target):
      pred = make_prediction(target)
      return Leaf(pred)

    if isEmpty(training):
      pred = make_prediction(target)
      return Leaf(pred)
    
    (ig, attr_name, vals, is_numeric) = self._best_split(target, training)

    best_attr = training[attr_name]
    #remove column from training data
    training = training.drop(attr_name, axis=1)

    target_splits = self._make_split(target, best_attr, vals, is_numeric)
    node = Node(ig, attr_name)
    for split in target_splits:
      branch = Branch(split['val'], split['exp'])
      child = self._train_tree(split['data'], training, depth - 1)
      if child == None:
        continue
      branch.child = child
      node.addBranch(branch)
    return node
  
  def _best_split(self, target, data):
    col_list = data.columns
    best_ig = 0
    best_val = None
    best_is_numeric = None
    best_arg_name = None
    for col in col_list:
      attribute = data[col]
      (ig, val, is_numeric) = information_gain(target, attribute)
      if ig > best_ig:
        best_ig = ig
        best_val = val
        best_is_numeric = is_numeric
        best_arg_name = col

    return (best_ig, best_arg_name, best_val, best_is_numeric)
  
  def _make_split(self, target, attribute, val, is_numeric):
    data_splits = []

    if is_numeric:
      target_lt = split(target, attribute, val, lt)
      target_gte = split(target, attribute, val, gte)
      data_splits.append({'data': target_lt, 'exp': lt, 'val': val})
      data_splits.append({'data': target_gte, 'exp': gte, 'val': val})

    else:
      for cat in val:
        target_split = split(target, attribute, cat, eq)
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