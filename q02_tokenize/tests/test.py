import unittest
from inspect import getfullargspec
import warnings
# warnings.filterwarnings("ignore")
from ..build import q02_tokenize
from greyatomlib.nlp_day_01_project.q02_tokenize.build import q02_tokenize as act_solution
import dill
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal

path = 'data/20news-bydate-train/'
student_return = q02_tokenize(path)
original_return = act_solution(path)


class Testing(unittest.TestCase):
    #  Check the arguements of the function

    def test_args(self):
        # Input parameters tests
        args = getfullargspec(q02_tokenize)
        self.assertEqual(len(args[0]), 1, "Expected arguments %d, Given %d" % (1, len(args[0])))

    def test_defaults(self):
        args = getfullargspec(q02_tokenize)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    def test_return_dataframe(self):
        assert_series_equal(student_return, original_return,
                            obj="The return values do not match expected values")
