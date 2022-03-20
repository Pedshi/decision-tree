import unittest
import pandas as pd
from cart.carttree import CARTTree


class TestCARTTree(unittest.TestCase):
    def test_TwoAttributesNoMaxDepth_PredictB(self):
        attribute_data = {'col1': [1, 2, 1, 1, 1], 'col2': ['x', 'y', 'x', 'y', 'x']}
        train_x = pd.DataFrame(data=attribute_data)
        target = pd.Series(data=['a', 'a', 'b', 'b', 'b'])
        tree = CARTTree()
        tree.train(train_x, target)
        predict_data = {'col1': [1], 'col2': ['x']}
        predict_x = pd.DataFrame(data=predict_data)
        expected = 'b'

        prediction = tree.predict(predict_x)

        self.assertEqual(expected, prediction)

    def test_UnanimousAnswer_PredictA(self):
        attribute_data = {'col1': [1, 2, 1, 1, 1], 'col2': ['x', 'y', 'x', 'y', 'x']}
        train_x = pd.DataFrame(data=attribute_data)
        target = pd.Series(data=['a', 'a', 'b', 'b', 'b'])
        tree = CARTTree()
        tree.train(train_x, target)
        predict_x = pd.DataFrame(data={'col1': [3], 'col2': ['x']})
        expected = 'a'

        prediction = tree.predict(predict_x)

        self.assertEqual(expected, prediction)


class TestBestSplit(unittest.TestCase):
    def test_CategoricalArgIsBest_ReturnCorrectCol2(self):
        attribute_data = {'col1': [1, 2, 1, 2, 1], 'col2': ['x', 'y', 'x', 'x', 'x']}
        target_data = ['a', 'a', 'b', 'b', 'b']
        df = pd.DataFrame(data=attribute_data)
        target = pd.Series(data=target_data)
        tree = CARTTree()
        expected = 'col2'

        (_, arg_name, _, _) = tree._best_split(df, target)

        self.assertEqual(expected, arg_name)

    def test_NumericArgIsBest_ReturnCol1(self):
        attribute_data = {'col1': [1, 2, 1, 1, 1], 'col2': ['x', 'y', 'x', 'y', 'x']}
        df = pd.DataFrame(data=attribute_data)
        target = pd.Series(data=['a', 'a', 'b', 'b', 'b'])
        tree = CARTTree()
        expected = 1

        (_, _, val, _) = tree._best_split(df, target)

        self.assertEqual(expected, val)
