import unittest
import os
import numpy as np
import golem

class Testroc(unittest.TestCase):
  def setUp(self):
    self.d = golem.data.gaussian_dataset([100, 100])

  def test_roc(self):
    d = self.d
    TPs, FPs = golem.helpers.roc(d.xs[:,0], d.ys[:,0])
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
    t0, f0 = golem.helpers.roc(scores, labels)
    t1, f1 = golem.helpers.roc(scores[::-1], labels[::-1])

    self.assert_((t0 == t1).all())
    self.assert_((f0 == f1).all())

  def test_plot(self):
    d = golem.DataSet(np.round(self.d.xs, 1), default=self.d)
    golem.plots.plot_roc(d, os.path.join('tests', 'plots', 'roc.eps'))
