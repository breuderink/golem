import numpy as np
from helpers import *
import algorithms as alg
from crossval import *
import loss

import unittest

class TestOneVsRest(unittest.TestCase):
  def testOVR(self):
    '''Test OneVsRest with a 4 class SVM'''
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    # Construct a *very* predictable DataSet with 4 classes
    d = DataSet(ys[:, :-1], ys)
    self.assert_(d.nclasses == 4)
    
    # Cross-validate and test for perfect classification
    cl = alg.OneVsRest(alg.SupportVectorMachine(sign_output=False))
    accs = [loss.accuracy(r) for r in 
      cross_validate(stratified_split(d, 2), cl)]
    self.assert_(np.mean(accs) == 1)

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestOneVsRest))
  return suite

if __name__ == '__main__':
  unittest.main()
