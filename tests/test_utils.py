import unittest
import pandas as pd
from tree.util import *


class TestEntropy(unittest.TestCase):
    def test_whenSeriesWithData_returnsCorrectEntropy(self):
        d = ['a', 'b', 'b', 'b']
        ser = pd.Series(data=d)
        expected = 0.8113

        response = entropy(ser)

        self.assertEqual(expected, response)


class TestInformationGainCat(unittest.TestCase):
    def test_whenTargetAndAttributeDefined_returnsCorrectInformationGain(self):
        parent_data = ['a', 'a', 'b', 'b']
        attribute_data = ['x', 'x', 'z', 'x']
        parent_ser = pd.Series(data=parent_data)
        attribute_ser = pd.Series(data=attribute_data)
        expected = 0.3113

        (ig, _) = information_gain_cat(parent_ser, attribute_ser)

        self.assertEqual(expected, ig)


class TestInformationGainNum(unittest.TestCase):
    def test_whenTargetAndAttributeDefined_returnsCorrectInformationGain(self):
        parent_data = ['a', 'a', 'b', 'b']
        attribute_data = [2, 2, 1, 2]
        parent_ser = pd.Series(data=parent_data)
        attribute_ser = pd.Series(data=attribute_data)
        expected = 0.3113

        (ig, _) = information_gain_num(parent_ser, attribute_ser)

        self.assertEqual(expected, ig)


class TestInformationGain(unittest.TestCase):
    def test_whenNumericAttribute_returns2AsBestValue(self):
        parent_data = ['a', 'a', 'b', 'b']
        attribute_data = [2, 2, 1, 3]
        parent_ser = pd.Series(data=parent_data)
        attribute_ser = pd.Series(data=attribute_data)
        expected = 2

        (_, val, _) = information_gain(parent_ser, attribute_ser)

        self.assertEqual(expected, val)

    def test_whenNumericAttribute_returnsCorrectIg(self):
        parent_data = ['a', 'a', 'b', 'b']
        attribute_data = [2, 2, 1, 3]
        parent_ser = pd.Series(data=parent_data)
        attribute_ser = pd.Series(data=attribute_data)
        expected = 0.3113

        (ig, _, _) = information_gain(parent_ser, attribute_ser)

        self.assertEqual(expected, ig)

    def test_whenCategoricAttribute_returnsCorrectIg(self):
        parent_data = ['a','a','b','b','b']
        attribute_data = ['x','z','x','z','z']
        parent_ser = pd.Series(data=parent_data)
        attribute_ser = pd.Series(data=attribute_data)
        expected = 0.02

        (ig, _, _) = information_gain(parent_ser, attribute_ser)

        self.assertEqual(expected, ig)


class TestGiniImpurity(unittest.TestCase):
    def test_whenNumericValues_returnsCorrectGini(self):
        data = [1, 2, 2, 2, 2, 2]
        ser = pd.Series(data=data)
        expected = 0.2778

        G = gini_impurity(ser)

        self.assertEqual(expected, G)

    def test_whenCategoricValues_returnsCorrectGini(self):
        data = ['a', 'b', 'b', 'c', 'c']
        ser = pd.Series(data=data)
        expected = 0.64

        gini = gini_impurity(ser)

        self.assertEqual(expected, gini)


if __name__ == '__main__':
    unittest.main()