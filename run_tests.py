from operator import mod
import unittest
import tests as test_cases

if __name__ == '__main__':
  runner = unittest.TextTestRunner()
  loader = unittest.defaultTestLoader

  suite = loader.loadTestsFromModule(test_cases)

  runner.run(suite)