import unittest
import numpy as np
from .. import DataSet
from ..nodes import BaseNode

class TestBaseNode(unittest.TestCase):
  def setUp(self):
    xs = np.random.rand(10, 4)
    ys = np.ones((10, 2))
    self.d = DataSet(xs, ys)
    self.n = BaseNode()

  def test_same_train_test(self):
    d = self.d
    n = self.n
    n.train(d)
    n.test(d)

    n.train(d)
    self.assertRaises(ValueError, n.test, DataSet(feat_shape=(2, 2), 
      default=d))

  def test_logger(self):
    n = BaseNode(name='testlog') 
    self.assertEqual(n.log.name, 'nodes.testlog')

  def test_assert_two_class(self):
    d = self.d
    d3 = DataSet(ys=np.ones((10, 3)), cl_lab=['a','b','c'], default=d)
    n = self.n
    n.assert_two_class(self.d)
    self.assertRaises(ValueError, n.assert_two_class, d3)
