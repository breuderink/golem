import unittest
import numpy as np
from helpers import artificialdata
from crossval import *

class TestCrossValidation(unittest.TestCase):
  def testStratifiedSplit(self):
    d = artificialdata.gaussian_dataset([30, 20, 10])
    d = d[np.lexsort(d.xs.T)] # order instances for comparison

    subsets = stratified_split(d, 10)
    self.assert_(len(subsets) == 10)
    for s in subsets:
      self.assert_(s.ninstances_per_class == [3, 2, 1])

    d2 = reduce(lambda a, b : a + b, subsets)
    self.assert_(d == d2[np.lexsort(d2.xs.T)])

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestCrossValidation))
  return suite

