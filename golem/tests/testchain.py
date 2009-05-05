import unittest
import numpy as np
from ..nodes import Chain, RationedChain
from .. import DataSet

class AddSumNode:
  def __init__(self):
    self.train_calls = 0
    self.test_calls = 0
    self.sum = None

  def train(self, d):
    self.sum = np.sum(d.xs)
    self.train_calls += 1

  def test(self, d):
    self.test_calls += 1
    return DataSet(xs=d.xs + self.sum, default=d)

class TestChain(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.ones((10, 1)), ys=np.ones((10, 1)))

  def test_chain(self):
    nodes = [AddSumNode() for n in range(3)]
    d = self.d
    c = Chain(nodes)
    c.train(d)
    np.testing.assert_equal([n.train_calls for n in nodes], [1, 1, 1])
    np.testing.assert_equal([n.test_calls for n in nodes], [1, 1, 0])
    np.testing.assert_equal([n.sum for n in nodes], 
      [10, (1 + 10) * 10, (1 + 10 + 110) * 10])

    np.testing.assert_equal(c.test(d).xs, 
      1 + 10 + (1 + 10) * 10 + (1 + 10 + (1 + 10) * 10) * 10 * np.ones((10, 1)))


class TestRationedChain(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(xs=np.ones((12, 1)), ys=np.ones((12, 1)))

  def test_chain(self):
    nodes = [AddSumNode() for n in range(3)]
    d = self.d
    c = RationedChain([1, 1, 2], nodes)
    c.train(d)
    np.testing.assert_equal([n.train_calls for n in nodes], [1, 1, 1])
    np.testing.assert_equal([n.test_calls for n in nodes], [2, 1, 0])
    np.testing.assert_equal([n.sum for n in nodes], 
      [1 * 3, (3 + 1) * 3, (12 + 3 + 1) * 6])
    np.testing.assert_equal(c.test(d).xs, 
      (12 + 3 + 1) * 6 + 12 + 3 + 1 * np.ones((12, 1)))
