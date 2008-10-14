import unittest
import operator
import numpy as np

from dataset import *

class TestDatasetConstruction(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction_list(self):
    '''Test the construction from a list with features and a list with
    labels.

    '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)
    self.assertEqual(d.nclasses, 2)
    self.assert_((d.xs == xs).all())
    self.assert_((d.ys == ys).all())
    
  def test_construction_unequal_ninstances(self):
    '''Raise if number of instances does not match '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([0, 1, 2]).reshape(3, 1)
    self.assertRaises(ValueError, DataSet, xs, ys);
  
  def test_construction_dims(self):
    '''Raise if number of instances does not match '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([0, 1]).reshape(2, 1)
    self.assertRaises(ValueError, DataSet, xs.flatten(), ys);
    self.assertRaises(ValueError, DataSet, xs, ys.flatten());
    
class TestDataset(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    self.d = DataSet(xs, ys, feature_labels=['f1', 'f2', 'f3'], 
      class_labels=['A', 'B'], ids=[[3], [4]])
      
  def test_equality(self):
    d = self.d
    self.assert_(d == d)
    self.assert_(d == DataSet(d.xs, d.ys, d.ids, d.feature_labels, 
      d.class_labels))
    self.assert_(d == DataSet(d.xs.copy(), d.ys.copy(), d.ids.copy(), 
      d.feature_labels, d.class_labels))
    self.assert_(d <> 3)
    self.assert_(d <> DataSet(np.zeros((2, 3)), np.zeros((2, 2))))
    self.assert_(d <> DataSet(d.xs, d.ys, None, d.feature_labels, 
      d.class_labels))

  def test_indexing(self):
    '''Test the indexing of DataSet.'''
    d = self.d
    self.assert_(d[:] == d)
    self.assert_(d[0] + d[1] == d)
    self.assert_(d[:-1] == d[0])
    self.assert_(d[0].nclasses == 2)
    self.assert_(d[0].nfeatures == 3)
    self.assert_(d[-1] == d[d.nclasses - 1])

    indices = np.arange(d.ninstances)
    self.assert_(d[indices==1] == d[1])

  def test_class_extraction(self):
    '''Test the extraction of a single class from DataSet'''
    d = self.d
    dA = d.get_class(0)
    dB = d.get_class(1)
    self.assert_(dA == d[1])
    self.assert_(dB == d[0])

  def test_sort(self):
    '''Test sorting the DataSet'''
    d = self.d
    ds = d.sort()
    # Test if ids are sorted
    self.assert_((np.sort(ds.ids.flatten()) == ds.ids.flatten()).all())
    # Test if dataset is the same but differently ordered
    self.assert_(d[np.lexsort(d.xs.T)] == ds[np.lexsort(ds.xs.T)])

 
  def test_iter(self):
    '''Test the iterator of DataSet.'''
    instances = [(x, y) for (x, y) in self.d]
    for i in xrange(len(instances)):
      (x, y) = instances[i]
      self.assert_((x == self.d.xs[i,:]).all())
      self.assert_((y == self.d.ys[i,:]).all())
  
  def test_str(self):
    '''Test string representation.'''
    self.assertEqual('DataSet with 2 instances, 3 features, 2 classes: [1, 1]',
      str(self.d))

  def test_add(self):
    '''Test the creation of compound datasets using the add-operator.'''
    d1 = self.d
    d2 = DataSet(np.array([[3, 3, 3], [4, 4, 4]]), np.array([[0, 1], [1, 0]]), 
      feature_labels=d1.feature_labels, class_labels=d1.class_labels)
    d3 = d1 + d2
    self.assert_(d1.nfeatures == d2.nfeatures)
    self.assert_(d1.nclasses == d2.nclasses)
    self.assert_((np.vstack([d1.xs, d2.xs]) == d3.xs).all())
    self.assert_((np.vstack([d1.ys, d2.ys]) == d3.ys).all())

    d4 = DataSet(np.array([[3, 3], [4, 4]]), np.array([[0, 1], [1, 0]]))
    self.assert_(d1.nfeatures <> d4.nfeatures)
    self.assertRaises(ValueError, DataSet.__add__, d1, d4)
    
    d5 = DataSet(np.array([[3, 3, 3], [4, 4, 4]]), 
      np.array([[0, 1, 0], [1, 0, 0]]))
    
    self.assert_(d1.nclasses <> d5.nclasses)
    self.assertRaises(ValueError, DataSet.__add__, d1, d5)
   
def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestDatasetConstruction))
  suite.addTest(unittest.makeSuite(TestDataset))
  return suite
