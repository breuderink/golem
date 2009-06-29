import unittest
import numpy as np
from .. import DataSet, helpers, nodes, cv, loss
from ..nodes.ensemble import bagging_splitter

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
      cv.cross_validate(cv.strat_splits(d, 2), cl)]
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
      cv.cross_validate(cv.strat_splits(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)

class TestBagging(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable DataSet with 4 classes
    ys = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(xs=ys[:, :-1], ys=ys)

  def test_baggging_split(self):
    d = self.d
    bs = bagging_splitter(d)
    for i in range(10):
      di = bs.next()
      self.assertEqual(di.ninstances, d.ninstances)
      self.assertEqual(di.nclasses, d.nclasses)
      self.assertNotEqual(di, d)

  def test_bagging(self):
    d = self.d
    self.assertEqual(d.nclasses, 4)
    wcl = nodes.WeakClassifier()
    wcl.train(d)
    bcl = nodes.Bagging(nodes.WeakClassifier(), 20)
    bcl.train(d)
    self.assert_(loss.accuracy(wcl.test(d)) < loss.accuracy(bcl.test(d)))
