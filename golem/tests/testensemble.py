import unittest
import numpy as np
from .. import DataSet, helpers, nodes, crossval, loss

class TestOneVsOne(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable DataSet with 4 classes
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(xs=ys[:, :-1], ys=ys)
    
  def testOVO(self):
    '''Test OneVsOne with a 4 class SVM'''
    d = self.d
    self.assertEqual(d.nclasses, 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsOne(nodes.SVM())
    accs = [loss.accuracy(r) for r in 
      crossval.cross_validate(crossval.stratified_split(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)

class TestOneVsRest(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable DataSet with 4 classes
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(xs=ys[:, :-1], ys=ys)

  def testOVR(self):
    '''Test OneVsRest with a 4 class SVM'''
    d = self.d
    self.assertEqual(d.nclasses, 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsRest(nodes.SVM())
    accs = [loss.accuracy(r) for r in 
      crossval.cross_validate(crossval.stratified_split(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)