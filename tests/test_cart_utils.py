import unittest
from cart.cart_utils import *
import pandas as pd


class TestGiniImpurity(unittest.TestCase):
    def test_NumericValues_ReturnsCorrectGini(self):
        data = [1, 2, 2, 2, 2, 2]
        ser = pd.Series(data=data)
        expected = 0.2778

        G = gini_impurity(ser)

        self.assertEqual(expected, G)

    def test_CategoricValues_ReturnsCorrectGini(self):
        data = ['a', 'b', 'b']
        ser = pd.Series(data=data)
        expected = 0.4444

        gini = gini_impurity(ser)

        self.assertEqual(expected, gini)


class TestLjoinFilter(unittest.TestCase):
    def test_GreaterThanTwoNumeric_ReturnsCorrectSplit(self):
        attribute = pd.Series(data=[1, 2, 3, 3])
        target = pd.Series(data=['a', 'a', 'b', 'a'])
        expected = pd.Series(data=['b', 'a'], index=[2, 3])

        result = ljoin_filter(target, attribute, 2, gt)

        self.assertTrue(expected.equals(result))


class TestGiniIndex(unittest.TestCase):
    def test_NumericAttribute_Returns2AsBestValue(self):
        attribute = pd.Series(data=[2, 2, 1, 3])
        target = pd.Series(data=['a', 'a', 'b', 'b'])
        expected = 0.6667

        gini = gini_index(target, attribute, 2, lte, gt)

        self.assertEqual(expected, gini)

    def test_CategoricAttribute_ReturnsCorrectIg(self):
        attribute = pd.Series(data=['x', 'z', 'x', 'z', 'z'])
        target = pd.Series(data=['a', 'a', 'b', 'b', 'b'])
        expected = 0.5334

        gini = gini_index(target, attribute, 'x', eq, neq)

        self.assertEqual(expected, gini)
