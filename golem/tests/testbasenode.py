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

  def test_existing_methods(self):
    class MaskTrain(BaseNode):
      def train(self, d):
        pass

    class MaskTest(BaseNode):
      def test(self, d):
        pass
    self.assertRaises(Exception, MaskTrain)
    self.assertRaises(Exception, MaskTest)

  def test_same_train_test(self):
    d = self.d
    n = self.n
    n.train(d)
    n.test(d) # no exception

    n.train(d)
    self.assertRaises(ValueError, n.test, DataSet(feat_shape=(2, 2), 
      default=d))

  def test_logger(self):
    class TestNode(BaseNode):
      pass

    n = BaseNode() 
    self.assertEqual(n.log.name, 'golem.nodes.BaseNode')

    tn = TestNode()
    self.assertEqual(tn.log.name, 'golem.nodes.TestNode')

  def test_assert_two_class(self):
    d = self.d
    d3 = DataSet(ys=np.ones((10, 3)), cl_lab=['a','b','c'], default=d)
    n = self.n
    n.assert_two_class(self.d)
    self.assertRaises(ValueError, n.assert_two_class, d3)
