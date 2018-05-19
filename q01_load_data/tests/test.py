import unittest
from inspect import getfullargspec
from ..build import q01_load_data
import dill
import pandas as pd
from greyatomlib.nlp_day_01_project.q01_load_data.build import q01_load_data as act_sol
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

path = 'data/20news-bydate-train/'
student_return = q01_load_data(path,seed=9)
original_return = act_sol(path,seed=9)

class Testing(unittest.TestCase):
    def test_args(self):

        # Input parameters tests
        args = getfullargspec(q01_load_data)
        self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))

    def test_defaults(self):
        args = getfullargspec(q01_load_data)
        self.assertEqual(args[3], (9,), "Expected default values do not match given default values")


    def test_return_dataframe(self):
        assert_series_equal(pd.Series(student_return[0]), pd.Series(original_return[0]),
                            obj="The return values do not match expected values")

    def test_return_1(self):
        self.assertListEqual(student_return[1], original_return[1],
                             "The return values do not match expected values")

    def test_return_2(self):
        self.assertListEqual(student_return[2], original_return[2],
                             "The return values do not match expected values")

    def test_return_3(self):
        assert_array_equal(student_return[3], original_return[3],
                           "The return values do not match expected values")

    def test_return_4(self):
        assert_array_equal(student_return[4], original_return[4],
                           "The return values do not match expected values")

