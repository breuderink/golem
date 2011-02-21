import unittest
import numpy as np
from .. import DataSet, data
from ..nodes import ApplyOverInstances, ZScore

class TestApplyOverInstances(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(X=np.arange(100).reshape(-1, 10), Y=np.ones(10),
      feat_shape=(5, 2))

  def test_map(self):
    d = self.d
    d2 = ApplyOverInstances(lambda x: x * 2).apply(d)
    self.assertEqual(d, DataSet(X=d2.X/2, default=d))

  def test_less_features(self):
    d = self.d
    d2 = ApplyOverInstances(lambda x: np.mean(x.flat)).apply(d)
    np.testing.assert_equal(d2.X, np.atleast_2d(np.mean(d.X, axis=0)))

  def test_nd_feat(self):
    d = self.d
    d2 = ApplyOverInstances(lambda x: x[:3, :]).apply(d)
    np.testing.assert_equal(d2.ndX, d.ndX[:3])

class TestZScore(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([40, 40, 40])
    
  def test_zscore(self):
    '''Test ZScore properties'''
    z = ZScore()
    zd = z.train_apply(self.d, self.d)

    # test for mean==0 and std==1
    np.testing.assert_almost_equal(np.mean(zd.X, axis=1), 0)
    np.testing.assert_almost_equal(np.std(zd.X, axis=1), 1)

    # test inverse
    zd_inv_X = zd.X * z.std + z.mean
    np.testing.assert_almost_equal(zd_inv_X, self.d.X)
