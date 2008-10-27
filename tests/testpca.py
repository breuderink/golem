import unittest
import numpy as np
from dataset import *
import nodes

class TestPCA(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    xs = np.random.rand(100, 3)
    xs = np.hstack([xs, -xs]) # make correlated
    self.d = DataSet(xs, np.ones((100, 1)), None)

  def test_nocov(self):
    '''Test that the PCA-transformed features are uncorrelated'''
    d = self.d
    n = nodes.PCA()
    n.train(d)
    d2 = n.test(d)

    self.assert_(d2.nfeatures == d.nfeatures)
    self.assert_(d2.nclasses == d.nclasses)
    self.assert_(d2.ninstances == d.ninstances)

    cov = np.cov(d2.xs, rowvar=False)
    cov_target=np.diag(np.diag(cov)) # construct a diagonal uncorrelated cov
    self.assert_(((cov - cov_target) ** 2 < 1e-8).all())
