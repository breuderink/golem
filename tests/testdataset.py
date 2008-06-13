import unittest
import numpy as np

from dataset import *

class TestDatasetConstruction(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction_emtpy(self):
    '''Test the construction of an empty DataSet.'''
    d = DataSet()
    d.add_instance((0, 0, 0), 0)
    d.add_instance((1, 1, 1), 1)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)

  def test_construction_list(self):
    '''Test the construction from a list with features and a list with
    labels.

    '''
    xs = [(0, 0, 0), (1, 1, 1)]
    ys = [0, 1]
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)
    
  def test_construction_unequal(self):
    '''Test that the construction from lists fails for unequal sized feature and
    label lists.

    '''
    xs = [(0, 0, 0), (1, 1, 1)]
    ys = [0, 1, 2]
    self.assertRaises(ValueError, DataSet, (xs, ys));
    
  def test_construction_var_length(self):
    '''Test the construction of a Dataset with instances that vary in the
    number of features.
    
    '''
    xs = [(0, 0), (1, 1, 1)]
    ys = [0, 1]
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertRaises(VariableNumberOfFeaturesException, d.get_xs)
    self.assertRaises(VariableNumberOfFeaturesException, DataSet.nfeatures.fget, d)

  def test_construction_array(self):
    '''Test the construction of a DataSet works from a NumPy array.'''
    xs = np.random.random((100, 5))
    ys = np.round(np.random.random((100, 1)))
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 100)
    self.assertEqual(d.nfeatures, 5)

class TestDataset(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    self.d = d = DataSet()
    d.add_instance((0, 0, 0), 0)
    d.add_instance((1, 1, 1), 1)
      
  def test_iter(self):
    '''Test the iterator of DataSet.'''
    instances = [(x, y) for (x, y) in self.d]
    for i in xrange(len(instances)):
      (x, y) = instances[i]
      self.assert_(x == (i, i, i))  # features are converted to arrays
      self.assert_(y == i)          # labels are converted to arrays
  
  def test_ninstances(self):
    '''Test if the number of instances is reported correctly after adding
    instances.

    '''
    d = self.d
    n = d.ninstances
    for i in range(5):
      self.assertEqual(d.ninstances, n)
      d.add_instance((n, n, n), n)
      n += 1
  
  def test_labels(self):
    '''Test if the labels are reported correctly after adding instances.'''
    d = self.d
    self.assert_((d.labels == [0, 1]).all())
    d.add_instance((2, 2, 2), 1)
    self.assert_((d.labels == [0, 1]).all())
    d.add_instance((3, 3, 3), -1)
    self.assert_((d.labels == [-1, 0, 1]).all())
    d.add_instance((4, 4, 4), 10)
    self.assert_((d.labels == [-1, 0, 1, 10]).all())

  def test_str(self):
    '''Test string representation.'''
    dstr = str(self.d)
    self.assertEqual('DataSet (2 instances, 3 features, 2 unique labels)', dstr)

  def test_var_length(self):
    '''Test if a variable number of features is handled correctly.'''
    d = self.d
    d.add_instance((1, 2), 3)
    self.assert_(d.const_feature_len() == False)
    self.assertRaises(VariableNumberOfFeaturesException, d.get_xs) 
  
  def test_plus(self):
    '''Test the creation of compound datasets using the add-operator.'''
    d1 = self.d
    d2 = DataSet()
    d2.add_instance((3, 3, 3), 3)
    d2.add_instance((4, 4, 4), 4)

    d3 = d1 + d2
    self.assert_(d3.ninstances == d1.ninstances + d2.ninstances)
    d1_tups = [(x, y) for (x, y) in d1] 
    d2_tups = [(x, y) for (x, y) in d2] 
    d3_tups = [(x, y) for (x, y) in d3] 
    self.assertEqual(d3_tups, d1_tups + d2_tups)
   
def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestDatasetConstruction))
  suite.addTest(unittest.makeSuite(TestDataset))
  return suite
