import unittest
import os
import numpy as np
from numpy.testing import assert_equal
from .. import data, helpers, plots, DataSet

class TestRoc(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([100, 100])

  def test_roc(self):
    d = self.d
    TPs, FPs = helpers.roc(d.xs[:,0], d.ys[:,0])
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
    t0, f0 = helpers.roc(scores, labels)
    t1, f1 = helpers.roc(scores[::-1], labels[::-1])

    self.assert_((t0 == t1).all())
    self.assert_((f0 == f1).all())
  
  def test_known(self):
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = helpers.roc(scores, labels)

    self.assert_((t0 == [0, .25, 1, 1]).all())
    self.assert_((f0 == [0, 0, .75, 1]).all())

  def test_plot(self):
    d = DataSet(np.round(self.d.xs, 1), default=self.d)
    plots.plot_roc(d, os.path.join('tests', 'plots', 'roc.eps'))

class TestOneOfN(unittest.TestCase):
  def test_simple(self):
    # test construction with one class
    assert_equal(helpers.to_one_of_n([0, 0, 0, 0]), np.ones((4, 1)))
    assert_equal(helpers.to_one_of_n([1, 1, 1]), np.ones((3, 1)))

    # test construction with two classes, cols sorted
    ys2d_a = helpers.to_one_of_n([0, 1, 1])
    ys2d_b = helpers.to_one_of_n([1, 2, 0])
    assert_equal(ys2d_a, np.array([[1, 0], [0, 1], [0, 1]]))
    assert_equal(ys2d_b, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))

  def test_cols(self):
    ys = helpers.to_one_of_n([0, 1, 2], [2, 1, 0])
    assert_equal(ys, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

  def test_cols_non_existing(self):
    ys = helpers.to_one_of_n([0, 1, 2], [5, 6, 7])
    assert_equal(ys, np.zeros((3, 3)))
