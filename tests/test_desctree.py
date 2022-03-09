import unittest
import pandas as pd
from tree.desctree import DescisionTree, make_prediction

class TestDescisionTree(unittest.TestCase):
  def test_whenBothColsGiveSameTarget_returnsBAsPrediction(self):
    attribute_data = {'col1': [1,2,1,1,1], 'col2':['x','y','x','y','x']} 
    target_data = ['a','a','b','b','b']
    train_x = pd.DataFrame(data=attribute_data)
    target = pd.Series(data=target_data)
    tree = DescisionTree()
    tree.train(target, train_x)
    predict_data = {'col1': [1], 'col2':['x']}
    predict_x = pd.DataFrame(data=predict_data)
    expected = 'b'

    prediction = tree.predict(predict_x)

    self.assertEqual(expected, prediction)
  
  def test_whenBothColsGiveSameTarget_returnsAAsPrediction(self):
    attribute_data = {'col1': [1,2,1,1,1], 'col2':['x','y','x','y','x']} 
    target_data = ['a','a','b','b','b']
    train_x = pd.DataFrame(data=attribute_data)
    target = pd.Series(data=target_data)
    tree = DescisionTree()
    tree.train(target, train_x)
    predict_data = {'col1': [3], 'col2':['x']}
    predict_x = pd.DataFrame(data=predict_data)
    expected = 'a'

    prediction = tree.predict(predict_x)

    self.assertEqual(expected, prediction)


class TestBestSplit(unittest.TestCase):
  def test_whenSecondArgIsBest_returnsCorrectArgName(self):
    attribute_data = {'col1': [1,2,1,2,1], 'col2':['x','y','x','x','x']} 
    target_data = ['a','a','b','b','b']
    df = pd.DataFrame(data=attribute_data)
    target = pd.Series(data=target_data)
    tree = DescisionTree()
    expected = 'col2'

    (_, arg_name, _, _) =  tree._best_split(target, df)

    self.assertEqual(expected, arg_name)

  def test_whenNumericArgIsBest_returnsCorrectValue(self):
    attribute_data = {'col1': [1,2,1,1,1], 'col2':['x','y','x','y','x']} 
    target_data = ['a','a','b','b','b']
    df = pd.DataFrame(data=attribute_data)
    target = pd.Series(data=target_data)
    tree = DescisionTree()
    expected = 2

    (_, _, val, _) =  tree._best_split(target, df)

    self.assertEqual(expected, val)

class TestMakeSplitNum(unittest.TestCase):
  def test_whenSplitOnAttribute2_returnsTargetsWhereAttributeIs2(self):
    target_data = ['a','a','b','b','b']
    attribute_data = [1,2,2,2,2]
    target = pd.Series(data=target_data)
    attribute = pd.Series(data=attribute_data)
    tree = DescisionTree()
    expected = pd.Series(data=['a','b','b','b'], index=[1,2,3,4])

    data_splits = tree._make_split(target, attribute, 2, True)
    new_target = data_splits[1]['data']

    self.assertTrue(expected.equals(new_target))

class TestMakeSplitCat(unittest.TestCase):
  def test_whenSplitOnAttributeX_returnsTargetsWhereAttributeIsX(self):
    target_data = ['a','a','b','b','b']
    attribute_data = ['x','y','z','x','x']
    target = pd.Series(data=target_data)
    attribute = pd.Series(data=attribute_data)
    vals = attribute.unique()
    tree = DescisionTree()
    expected = pd.Series(data=['a','b','b'], index=[0,3,4])

    data_splits = tree._make_split(target, attribute, vals, False)
    new_target = data_splits[0]['data']

    self.assertTrue(expected.equals(new_target))

class TestMakePrediction(unittest.TestCase):
  def test_whenCategoryBIsDominant_returnsCatBAsPrediction(self):
    target_data = ['a','a','b','b','b']
    target = pd.Series(data=target_data)
    expected = 'b'

    prediction = make_prediction(target)

    self.assertEqual(expected, prediction)

  def test_whenNumericTarget_returnsAverageOfTarget(self):
    target_data = [2,1,2,2]
    target = pd.Series(data=target_data)
    expected = 1.75

    prediction = make_prediction(target)

    self.assertEqual(expected, prediction)