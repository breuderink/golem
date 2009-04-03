import unittest
import numpy as np
from ..import DataSet, data
from ..nodes import ZScore

class TestZScore(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    self.d = data.gaussian_dataset([40, 40, 40])
    
  def test_zscore(self):
    '''Test ZScore properties'''
    z = ZScore()
    z.train(self.d)
    zd = z.test(self.d)

    # test for mean==0 and std==1
    np.testing.assert_almost_equal(np.mean(zd.xs, axis=0), 
      np.zeros(self.d.nfeatures))
    np.testing.assert_almost_equal(np.std(zd.xs, axis=0), 
      np.ones(self.d.nfeatures))

    # test inverse
    zd_inv_xs = zd.xs * z.std + z.mean
    np.testing.assert_almost_equal(zd_inv_xs, self.d.xs)

  def test_broadcasting(self):
    '''Test ZScore's broadcasting behaviour'''
    # test with d.nfeatures == d.ninstances
    # broadcasting could change behaviour, test that it does not.
    xs = np.random.random((4, 4))
    d = DataSet(xs=xs, ys=np.ones((4, 1)))
    z = ZScore()
    z.train(d)
    zd = z.test(d)

    # test for mean==0 and std==1
    np.testing.assert_almost_equal(np.mean(zd.xs, axis=0), 
      np.zeros(d.nfeatures))
    np.testing.assert_almost_equal(np.std(zd.xs, axis=0), 
      np.ones(d.nfeatures))
