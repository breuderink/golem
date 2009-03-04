import unittest
import os.path
import numpy as np
from .. import DataSet, nodes, helpers, data, plots

class TestCSP(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    da = data.gaussian_dataset([100, 100])
    self.d = DataSet(np.hstack([da.xs, np.random.random(da.xs.shape)]), 
      da.ys, da.ids, feat_shape=[1, 4])
    self.n = nodes.CSP(m=2)

  def test_class_diag_descending(self):
    '''Test for diagonal, descending class cov'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d.get_class(0))

    self.assert_(d2.nfeatures == 2)
    self.assert_(d2.nclasses == 2)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    self.assert_((np.abs(np.diag(cov) - np.sort(np.diag(cov))[::-1]) \
      < 1e-8).all())
  
  def test_nocov(self):
    '''Test that the CSP-transformed features are uncorrelated'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)

    self.assert_(d2.nfeatures == 2)
    self.assert_(d2.nclasses == 2)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    self.assert_(np.sum(np.abs(cov - np.eye(cov.shape[0]))) < 1e-8)
  
  def test_plot(self):
    '''Plot CSP for visual inspection'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)
    plots.scatter_plot(d2, os.path.join('tests', 'plots', 'csp.eps'))
  
  def test_2d(self):
    '''Test CSP on 2D-shaped features'''
    d, n = self.d, self.n
    n.train(d)
    d2 = DataSet(d.xs.reshape(-1, 8), d.ys[::2], d.ids[::2], feat_shape=[2, 4])
    n2 = nodes.CSP(m=2)
    n2.train(d2)
    d2 = n2.test(d2)
    self.assert_(d2.nfeatures == 4)
    self.assert_(d2.nclasses == 2)

    cov = np.cov(np.concatenate(d2.nd_xs, axis=0), rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    self.assert_(np.sum(np.abs(cov - np.eye(cov.shape[0]))) < 1e-8)
