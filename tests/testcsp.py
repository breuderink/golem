import unittest
import os.path
import numpy as np
from golem import DataSet, nodes, helpers, data, plots

class TestCSP(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    da = data.gaussian_dataset([200, 200])
    self.d = DataSet(np.hstack([da.xs, -da.xs]), da.ys, da.ids)
    self.n = nodes.CSP(m=2)

  def test_class_diag(self):
    '''Test for diagonal class cov'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d.get_class(0))

    self.assert_(d2.nfeatures == 2) # rank(cov) == 2
    self.assert_(d2.nclasses == 2)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
  
  def test_nocov(self):
    '''Test that the CSP-transformed features are uncorrelated'''
    d, n = self.d, self.n
    n.train(d)
    d2 = n.test(d)

    self.assert_(d2.nfeatures == 2) # rank(cov) == 2
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
