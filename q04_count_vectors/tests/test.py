import unittest
from inspect import getfullargspec
import warnings
# warnings.filterwarnings("ignore")
from ..build import q04_count_vectors
from numpy.testing import assert_array_equal
path = 'data/20news-bydate-train/'
a,b = q04_count_vectors(path)

class Testing(unittest.TestCase):

    #  Check the arguements of the function
    def test_args(self):
        # Input parameters tests
        args = getfullargspec(q04_count_vectors)
        self.assertEqual(len(args[0]), 4, "Expected arguments %d, Given %d" % (1, len(args[0])))

        # check the defaults of the function

    def test_defaults(self):
        args = getfullargspec(q04_count_vectors)
        self.assertEqual(args[3], ((1, 2),0.5,2), "Expected default values do not match given default values")


    def test_return_4(self):
        self.assertEqual(a.shape,(2262, 72601),
                         "The return values do not match expected values")

    def test_return_4_2(self):
        self.assertEqual(b.shape,(9052, 72601),
                         "The return values do not match expected values")
