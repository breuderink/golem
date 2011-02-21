import unittest
import numpy as np
from .. import DataSet, nodes
from ..stat import lw_cov

class TestPCA(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    xs = np.random.rand(100, 3)
    xs = np.hstack([xs, -xs, np.zeros((100, 0))]) # make correlated
    self.d = DataSet(xs=xs, ys=np.ones((100, 1)))

  def test_nocov_descending(self):
    '''Test that the PCA-transformed features are uncorrelated, and ordered'''
    d = self.d
    n = nodes.PCA(ndims=d.nfeatures)

    n.train(d)
    d2 = n.apply(d)

    self.assertEqual(d2.nfeatures, d.nfeatures)
    self.assertEqual(d2.nclasses, d.nclasses)
    self.assertEqual(d2.ninstances, d.ninstances)

    cov = lw_cov(d2.X)
    np.testing.assert_almost_equal(np.diag(cov), np.sort(np.diag(cov))[::-1])

  def test_ndims(self):
    '''Test PCA with given dimensionality'''
    d = self.d
    for td in range(0, d.nfeatures + 1):
      # Build PCA
      p = nodes.PCA(ndims=td)
      self.assertEqual(str(p), 'PCA')
      p.train(d)
      self.assertEqual(str(p), 'PCA (6D -> %dD)' % td)
      d2 = p.apply(d)

      self.assertEqual(d2.nfeatures, td)
      if td > 1:
        cov = lw_cov(d2.X)
        self.assertAlmostEqual(np.trace(cov), np.sum(cov))

  def test_retain(self):
    '''Test that the PCA retains minimum but at least the specified variance'''
    d = self.d
    # Test for different amounts of retaind variance
    for retain in np.linspace(0, 1, 10):
      # Build PCA
      n = nodes.PCA(retain=retain)
      d2 = n.train_apply(d, d)

      retained = np.trace(lw_cov(d2.X)) / np.trace(lw_cov(d.X))
      # test that we retain enough variance with limited precision:
      self.assert_(retain < retained + 1e-8, 
        'retain=%.6g, retained=%.6g' % (retain, retained))

      # Test that we do not have too many PCs
      if d2.nfeatures > 1:
        one_less = nodes.PCA(ndims=d2.nfeatures-1)
        d3 = one_less.train_apply(d, d)
        self.assertEqual(d3.nfeatures, d2.nfeatures - 1)
        
        retained_ol = np.trace(lw_cov(d3.X)) / np.trace(lw_cov(d.X))
        # test that a lower dimensional project retains not enough variance
        # with limited precision:
        self.assert_(retain >= retained_ol - 1e-8, 
          'retain=%.6g, retained_ol=%.6g' % (retain, retained_ol))
