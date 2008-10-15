import unittest
import numpy as np
from dataset import *
import algorithms as alg

class TestZScore(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    self.d = helpers.artificialdata.gaussian_dataset([40, 40, 40])
    
  def test_zscore(self):
    z = alg.ZScore()
    z.train(self.d)
    zd = z.test(self.d)

    # Test for mean==0 and std==1
    self.assert_((np.abs(np.mean(zd.xs, axis=0)) < 1e-8).all())
    self.assert_((np.abs(np.std(zd.xs, axis=0) - 1) < 1e-8).all())

    # Test inverse
    zd_inv_xs = zd.xs * z.std + z.mean
    self.assert_((np.abs(zd_inv_xs - self.d.xs) < 1e-8).all())

  def test_broadcasting(self):
    # Test with d.nfeatures == d.ninstances
    # Broadcasting could change behaviour. Test that it does not.
    xs = np.random.random((4, 4))
    d = DataSet(xs, np.ones((4, 1)))
    z = alg.ZScore()
    z.train(d)
    zd = z.test(d)

    # Test for mean==0 and std==1
    self.assert_((np.abs(np.mean(zd.xs, axis=0)) < 1e-8).all())
    self.assert_((np.abs(np.std(zd.xs, axis=0) - 1) < 1e-8).all())
