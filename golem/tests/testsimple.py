import unittest
import numpy as np
from .. import DataSet
from ..nodes import FeatMap

class TestFeatMap(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.arange(100).reshape(10, -1), feat_shape=(5, 2), 
      ys=np.ones((10, 1)), feat_lab=['f%d' % i for i in range(10)], 
      feat_nd_lab=[['a', 'b', 'c', 'd', 'e'], ['A', 'B']])

  def test_map(self):
    d = self.d
    n = FeatMap(lambda x: x * 2)
    d2 = n.apply(d)
    d3 = DataSet(xs=d2.xs/2, default=d)
    self.assertEqual(d, d3)

  def test_less_features(self):
    d = self.d
    n = FeatMap(lambda x: np.mean(x.flat))
    d2 = n.apply(d)
    np.testing.assert_equal(d2.xs, (np.arange(10) * 10 + 4.5).reshape(10, -1))

  def test_nd_feat(self):
    d = self.d
    n = FeatMap(lambda x: x[:3, :])
    d2 = n.apply(d)
    self.assertEqual(d2.feat_shape, (3, 2))
    self.assertEqual(d2.ninstances, 10)
    np.testing.assert_equal(d2.xs, 
      np.arange(10).reshape(10, 1) * 10 + np.arange(6))
