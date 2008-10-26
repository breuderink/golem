import numpy as np
from helpers import *
import nodes
from crossval import *
import loss

import unittest

class TestOneVsOne(unittest.TestCase):
  def testOVO(self):
    '''Test OneVsOne with a 4 class SVM'''
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    # Construct a *very* predictable DataSet with 4 classes
    d = DataSet(ys[:, :-1], ys, None)
    self.assert_(d.nclasses == 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsOne(nodes.SVM())
    accs = [loss.accuracy(r) for r in 
      cross_validate(stratified_split(d, 2), cl)]
    self.assert_(np.mean(accs) == 1)

if __name__ == '__main__':
  unittest.main()
