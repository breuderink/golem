import unittest
import numpy as np
from .. import DataSet, helpers, nodes, loss, crossval

class TestOneVsRest(unittest.TestCase):
  def testOVR(self):
    '''Test OneVsRest with a 4 class SVM'''
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    # Construct a *very* predictable DataSet with 4 classes
    d = DataSet(xs=ys[:, :-1], ys=ys)
    self.assertEqual(d.nclasses, 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsRest(nodes.SVM())
    accs = [loss.accuracy(r) for r in 
      crossval.cross_validate(crossval.stratified_split(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)
