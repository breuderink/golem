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
    '''Test that masking an exisiting method raises an exception'''
    class MaskTrain(BaseNode):
      def train(self, d):
        pass

    class MaskApply(BaseNode):
      def apply(self, d):
        pass
    self.assertRaises(Exception, MaskTrain)
    self.assertRaises(Exception, MaskApply)

  def test_compatible_train_test(self):
    d = self.d
    n = self.n
    n.train(d)
    n.apply(d) # no exception

    n.train(d)
    self.assertRaises(ValueError, n.apply, DataSet(feat_shape=(2, 2), 
      default=d))

  def test_logger_name(self):
    class TestNode(BaseNode):
      pass

    n = BaseNode() 
    self.assertEqual(n.log.name, 'golem.nodes.BaseNode')

    tn = TestNode()
    self.assertEqual(tn.log.name, 'golem.nodes.TestNode')
