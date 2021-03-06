import unittest
import numpy as np
from ..nodes import Chain
from .. import DataSet

class AddSumNode:
  def __init__(self):
    self.train_calls = 0
    self.test_calls = 0
    self.sum = None

  def train(self, d):
    self.sum = np.sum(d.X)
    self.train_calls += 1

  def apply(self, d):
    self.test_calls += 1
    return DataSet(X=d.X + self.sum, default=d)

class TestChain(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(X=np.ones((1, 10)), Y=np.ones((1, 10)))

  def test_chain(self):
    d = self.d
    nodes = [AddSumNode() for n in range(3)]
    c = Chain(nodes)
    c.train(d)
    np.testing.assert_equal([n.train_calls for n in nodes], [1, 1, 1])
    np.testing.assert_equal([n.test_calls for n in nodes], [1, 1, 0])
    np.testing.assert_equal([n.sum for n in nodes], 
      [10, (1 + 10) * 10, (1 + 10 + 110) * 10])

    np.testing.assert_equal(c.apply(d).X, 
      1 + 10 + (1 + 10) * 10 + (1 + 10 + (1 + 10) * 10) * 10 * d.X)
