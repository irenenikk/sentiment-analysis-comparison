from unittest import TestCase
from science_utils import calculate_p_value, sign_test_p_value, sign_test_lists, swap_randomly
import numpy as np

class TestScienceUtils(TestCase):

    def test_calculating_p_value(self):
        # this is an example case from Siegel and Castellan (1986)
        N = 14
        k = 3
        self.assertEqual(0.0574, round(calculate_p_value(N, k, 0.5), 4))

    def test_calculating_p_value(self):
        # the p-value is calculated using https://www.graphpad.com/quickcalcs/binomial2/
        N = 10
        k = 2
        self.assertEqual(0.1094, round(calculate_p_value(N, k, 0.5), 4))

    def test_calculating_p_value_from_experiments(self):
        # this is an example case from Siegel and Castellan (1986)
        plus = 11
        minus = 3
        null = 3
        self.assertEqual(0.0574, round(sign_test_p_value(plus, minus, null), 4))

    def test_calculating_p_value_from_experiments(self):
        # this is an example case from Siegel and Castellan (1986)
        A = [5, 4, 6, 6, 3, 2, 5, 3, 1, 4, 5, 4, 4, 7, 5, 5, 5]
        B = [3, 3, 4, 5, 3, 3, 2, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1]
        self.assertEqual(0.0574, round(sign_test_lists(A, B), 4))

    def test_swapping_changes_correct_indices(self):
        A = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        B = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        shift_indices = [0, 2, 4, 9]
        A_, B_ = swap_randomly(A, B, shift_indices)
        self.assertListEqual(A_.tolist(), [1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
        self.assertListEqual(B_.tolist(), [0, 1, 0, 1, 0, 1, 1, 1, 1, 0])

    def test_swapping_changes_lists(self):
        A = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        B = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        A_, B_ = swap_randomly(A, B)
        # check that the arrays have changed
        self.assertFalse(np.all(np.equal(A_, B_)))        
