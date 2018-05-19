import unittest
from inspect import getfullargspec
import warnings
# warnings.filterwarnings("ignore")
from ..build import q03_stop_word_stemmer
from greyatomlib.nlp_day_01_project.q03_stop_word_stemmer.build import q03_stop_word_stemmer as act_solution
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))


path = 'data/20news-bydate-train/'
student_return = q03_stop_word_stemmer(path)
original_return = act_solution(path)

class Testing(unittest.TestCase):
    #  Check the arguements of the function
    def test_args(self):
        # Input parameters tests
        args = getfullargspec(q03_stop_word_stemmer)
        self.assertEqual(len(args[0]), 1, "Expected arguments %d, Given %d" % (1, len(args[0])))

    def test_defaults(self):
        args = getfullargspec(q03_stop_word_stemmer)
        self.assertEqual(args[3], None, "Expected default values do not match given default values")

    def test_return_1(self):
        self.assertListEqual(student_return, original_return,
                             "The return values do not match expected values")


