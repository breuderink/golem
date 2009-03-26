import unittest
import os
import numpy as np
from numpy.testing import assert_equal
from .. import data, plots, DataSet
from ..helpers import to_one_of_n, roc, auc, auc_confidence

class TestOneOfN(unittest.TestCase):
  def test_simple(self):
    # test construction with one class
    assert_equal(to_one_of_n([0, 0, 0, 0]), np.ones((4, 1)))
    assert_equal(to_one_of_n([1, 1, 1]), np.ones((3, 1)))

    # test construction with two classes, cols sorted
    ys2d_a = to_one_of_n([0, 1, 1])
    ys2d_b = to_one_of_n([1, 2, 0])
    assert_equal(ys2d_a, np.array([[1, 0], [0, 1], [0, 1]]))
    assert_equal(ys2d_b, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))

  def test_cols(self):
    ys = to_one_of_n([0, 1, 2], [2, 1, 0])
    assert_equal(ys, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

  def test_cols_non_existing(self):
    ys = to_one_of_n([0, 1, 2], [5, 6, 7])
    assert_equal(ys, np.zeros((3, 3)))

  def test_2d_input(self):
    ok_2d = to_one_of_n(np.atleast_2d([0, 1, 1]))
    assert_equal(ok_2d, np.array([[1, 0], [0, 1], [0, 1]]))

    self.assertRaises(AssertionError, to_one_of_n, np.ones((3, 3)))

class TestROC(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([100, 100])

  def test_roc(self):
    d = self.d
    TPs, FPs = roc(d.xs[:,0], d.ys[:,0])
    # test mononely increasing
    self.assert_((np.sort(TPs) == TPs).all())
    self.assert_((np.sort(FPs) == FPs).all())
    
    self.assert_(np.min(TPs) == 0)
    self.assert_(np.max(TPs) == 1)
    self.assert_(np.min(FPs) == 0)
    self.assert_(np.max(FPs) == 1)

  def test_stability(self):
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)
    t1, f1 = roc(scores[::-1], labels[::-1])

    self.assert_((t0 == t1).all())
    self.assert_((f0 == f1).all())
  
  def test_known(self):
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)

    self.assert_((t0 == [0, .25, 1, 1]).all())
    self.assert_((f0 == [0, 0, .75, 1]).all())

  def test_plot(self):
    d = DataSet(np.round(self.d.xs, 1), default=self.d)
    plots.plot_roc(d, 'roc.eps')

class TestAUC(unittest.TestCase):
  def test_AUC(self):
    self.assertEqual(auc([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]), 1)
    self.assertEqual(auc([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]), 0)
    self.assertEqual(auc([1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0]), .5)

    # test symmetry and bounds
    N = 100
    for rho in [.1, .3, .5]:
      for i in range(20):
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertAlmostEqual(auc(xs, ys), 1-auc(xs, np.abs(ys-1)))
        self.assert_(0 <= auc(xs, ys) <= 1)


  def test_AUC_confidence(self):
    # we do not know much, but we can test for trends
    self.assert_(auc_confidence(1) > auc_confidence(100))
    self.assert_(auc_confidence(100, rho=.1) > auc_confidence(100))
    self.assert_(auc_confidence(100, delta=1e-8) > auc_confidence(100))

    # and symmetry
    for rho in [.01, .1, .5]:
      self.assertAlmostEqual(auc_confidence(100, rho=rho),
        auc_confidence(100, rho=1-rho))

  def test_monte_carlo(self):
    SAMPLES = 100
    for N in [100, 1000]:
      for rho in [0.1, .5, .9]:
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertEqual(ys.mean(), rho)
        aucs = []
        for i in range(SAMPLES):
          np.random.shuffle(ys)
          aucs.append(auc(xs, ys))
        for delta in [.95, .05, .001, .0001]:
          dev = np.abs(np.array(aucs) - 0.5)
          epsilon = auc_confidence(N, rho, delta)
          e_epsilon = sorted(dev)[int(SAMPLES * (1-delta))]
          e_p = np.mean(dev > epsilon)
          self.assert_(e_p <= delta, 
            'empirical p (=%f) > delta (=%f)' % (e_p, delta))
