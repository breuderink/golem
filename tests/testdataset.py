import unittest
import numpy as np

from dataset import *

class TestDatasetConstruction(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction_emtpy(self):
    d = DataSet()
    d.add_instance((0, 0, 0), 0)
    d.add_instance((1, 1, 1), 1)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)

  def test_construction_list(self):
    xs = [(0, 0, 0), (1, 1, 1)]
    ys = [0, 1]
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)
    
  def test_construction_var_length(self):
    xs = [(0, 0), (1, 1, 1)]
    ys = [0, 1]
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertRaises(VariableNumberOfFeaturesException, d.get_xs)
    self.assertRaises(VariableNumberOfFeaturesException, DataSet.nfeatures.fget, d)

  def test_construction_array(self):
    xs = np.array([(0, 0, -1), (1, 1, -1)])
    ys = np.array([0, 1])
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)

class TestDatasetMisc(unittest.TestCase):
  def setUp(self):
    self.d = d = DataSet()
    d.add_instance((0, 0, 0), 0)
    d.add_instance((1, 1, 1), 1)
      
  def test_iter(self):
    instances = [(x, y) for (x, y) in self.d]
    for i in range(len(instances)):
      (tx, ty) = ((i, i, i), i)
      (x, y) = instances[i]
      self.assert_((x == tx).all())
      self.assert_((y == ty).all())
  
  def test_ninstances(self):
    d = self.d
    n = d.ninstances
    for i in range(5):
      self.assertEqual(d.ninstances, n)
      d.add_instance((n, n, n), n)
      n += 1
  
  def test_labels(self):
    d = self.d
    
    self.assert_((d.labels == [0, 1]).all())
    d.add_instance((2, 2, 2), 1)
    self.assert_((d.labels == [0, 1]).all())
    d.add_instance((3, 3, 3), -1)
    self.assert_((d.labels == [-1, 0, 1]).all())
    d.add_instance((4, 4, 4), 10)
    self.assert_((d.labels == [-1, 0, 1, 10]).all())

    
  
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDatasetConstruction))
    suite.addTest(unittest.makeSuite(TestDatasetMisc))
    return suite
