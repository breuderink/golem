import unittest
import numpy as np
from .. import DataSet, nodes

class TestPCA(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    xs = np.random.rand(100, 3)
    xs = np.hstack([xs, -xs, np.zeros((100, 0))]) # make correlated
    self.d = DataSet(xs, np.ones((100, 1)), None)

  def test_nocov_descending(self):
    '''Test that the PCA-transformed features are uncorrelated, and ordered'''
    d = self.d
    n = nodes.PCA()
    n.train(d)
    d2 = n.test(d)

    self.assert_(d2.nfeatures == d.nfeatures)
    self.assert_(d2.nclasses == d.nclasses)
    self.assert_(d2.ninstances == d.ninstances)

    cov = np.cov(d2.xs, rowvar=False)
    self.assertAlmostEqual(np.trace(cov), np.sum(cov))
    self.assert_((np.abs(np.diag(cov) - np.sort(np.diag(cov))[::-1]) \
      < 1e-8).all())
  
  def test_ndims(self):
    '''Test PCA with given dimensionality'''
    d = self.d
    for td in range(0, d.nfeatures + 1):
      # Build PCA
      p = nodes.PCA(ndims=td)
      p.train(d)
      d2 = p.test(d)

      self.assert_(d2.nfeatures==td)
      if td > 1:
        cov = np.cov(d2.xs, rowvar=False)
        self.assertAlmostEqual(np.trace(cov), np.sum(cov))


  def test_retain(self):
    '''Test that the PCA retains minimum but at least the specified variance'''
    d = self.d
    # Test for different amounts of retaind variance
    for retain_v in np.linspace(0, 1, 10):

      # Build PCA
      n = nodes.PCA(retain=retain_v)
      n.train(d)
      d2 = n.test(d)

      self.assert_(d2.nclasses == d.nclasses)
      self.assert_(d2.ninstances == d.ninstances)

      retained = np.sum(np.var(d2.xs, axis=0)) / np.sum(np.var(d.xs, axis=0))
      self.assert_(round(retained, 4) >= round(retain_v, 4))

      # Test that we do not have too many PCs
      if d2.nfeatures > 1:
        one_less = nodes.PCA(ndims=d2.nfeatures - 1)
        one_less.train(d)
        d3 = one_less.test(d)
        self.assert_(d3.nfeatures == d2.nfeatures - 1)
        retained_ol = np.sum(np.var(d3.xs, axis=0)) / \
          np.sum(np.var(d.xs, axis=0))
        self.assert_(retained_ol < retain_v)
