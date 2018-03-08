import unittest
from src.main import calculate_accuracy


class TestStringMethods(unittest.TestCase):

    def test_calculate_accuracy(self):
        actual = ['1', '2', '3']
        predict = ['1', '2', '4']
        self.assertAlmostEqual(2./3., calculate_accuracy(actual, predict), places=7)


if __name__ == '__main__':
    unittest.main()