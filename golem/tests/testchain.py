import unittest
import numpy as np
from ..nodes import Chain
from .. import DataSet

class AddMeanNode:
  def __init__(self):
    self.train_calls = 0
    self.mean = None

  def train(self, d):
    self.mean = np.mean(d.xs)
    self.train_calls += 1

  def test(self, d):
    return DataSet(xs=d.xs + self.mean, default=d)

class TestChain(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.ones((10, 2)), ys=np.ones((10, 1)))

  def testChain(self):
    nodes = [AddMeanNode() for n in range(3)]
    d = self.d
    c = Chain(nodes)
    c.train(d)
    np.testing.assert_equal([n.mean for n in nodes], [1, 2, 4])
    np.testing.assert_equal(c.test(d).xs, 8 * np.ones((10, 2)))

