import unittest
import numpy as np
from .. import DataSet, helpers, nodes, cv, perf
from ..nodes.ensemble import bagging_splitter

class TestOneVsOne(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable dataset with 4 classes
    Y = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(X=Y[:-1], Y=Y)
    
  def testOVO(self):
    '''Test OneVsOne with a 4 class SVM'''
    d = self.d
    self.assertEqual(d.nclasses, 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsOne(nodes.SVM(c=1e2))
    accs = [perf.accuracy(r) for r in 
      cv.cross_validate(cv.strat_splits(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)

class TestOneVsRest(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable dataset with 4 classes
    Y = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(X=Y[:-1], Y=Y)

  def testOVR(self):
    '''Test OneVsRest with a 4 class SVM'''
    d = self.d
    self.assertEqual(d.nclasses, 4)
    
    # Cross-validate and test for perfect classification
    cl = nodes.OneVsRest(nodes.SVM(c=1e2))
    accs = [perf.accuracy(r) for r in 
      cv.cross_validate(cv.strat_splits(d, 2), cl)]
    self.assertEqual(np.mean(accs), 1)

class TestBagging(unittest.TestCase):
  def setUp(self):
    # Construct a *very* predictable DataSet with 4 classes
    Y = helpers.to_one_of_n(np.arange(60) % 4)
    self.d = DataSet(X=Y[:-1], Y=Y)

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
    self.assert_(perf.accuracy(wcl.apply(d)) < perf.accuracy(bcl.apply(d)))
