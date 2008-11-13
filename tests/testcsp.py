import unittest
import os.path
import numpy as np
from golem import DataSet, nodes, helpers, data, plots

class TestCSP(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    da = data.gaussian_dataset([200, 200])
    self.d = DataSet(np.hstack([da.xs, -da.xs]), da.ys, da.ids)
    self.n = nodes.CSP(m=1)

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
  
  def test_2d(self):
    '''Test CSP on 2D-shaped features'''
    d, n = self.d, self.n
    n.train(d)
    print d.ys[::2].shape
    d2 = DataSet(d.xs.reshape(200, 8), d.ys[::2], d.ids[::2], feat_shape=[2, 4])
    n2 = nodes.CSP(m=2, axis=0)    
    n2.train(d2)
    d2 = n2.test(d2)
    self.assert_(d2.nfeatures == 4) # rank(cov) * 2 = 2 * 2
    self.assert_(d2.nclasses == 2)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    self.assert_(np.sum(np.abs(cov - np.eye(cov.shape[0]))) < 1e-8)

