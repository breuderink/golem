import unittest
import operator
import numpy as np

from golem import DataSet

class TestDataSetConstruction(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction_list(self):
    '''Test the construction from a list with features and a list with
    labels.
    '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    d = DataSet(xs, ys, None)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.nfeatures, 3)
    self.assertEqual(d.nclasses, 2)
    self.assert_((d.xs == xs).all())
    self.assert_((d.ys == ys).all())
    
  def test_construction_unequal_ninstances(self):
    '''Raise if number of instances does not match '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([0, 1, 2]).reshape(3, 1)
    self.assertRaises(ValueError, DataSet, xs, ys, None);
  
  def test_construction_dims(self):
    '''Raise if number of instances does not match '''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([0, 1]).reshape(2, 1)
    self.assertRaises(ValueError, DataSet, xs.flatten(), ys, None);
    self.assertRaises(ValueError, DataSet, xs, ys.flatten(), None);
    
class TestDataSet(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    ids = np.array([[3], [4]])
    self.d = DataSet(xs, ys, ids, feat_lab=['f1', 'f2', 'f3'], 
      cl_lab=['A', 'B'])
      
  def test_equality(self):
    d = self.d
    self.assert_(d == d)
    self.assert_(d == DataSet(d.xs, d.ys, d.ids, 
      feat_lab=d.feat_lab, cl_lab=d.cl_lab))
    self.assert_(d == DataSet(d.xs.copy(), d.ys.copy(), d.ids.copy(), 
      cl_lab=d.cl_lab, feat_lab=d.feat_lab))
    self.assert_(d <> 3)
    self.assert_(d <> DataSet(np.zeros((2, 3)), np.zeros((2, 2)), None))
    self.assert_(d <> DataSet(d.xs, d.ys, None, 
      feat_lab=d.feat_lab, cl_lab=d.cl_lab))

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

  def test_ndxs(self):
    '''Test multidimensional xs'''
    xs = np.arange(100).reshape(10, 10)
    ys = np.ones((10, 1))
    d = DataSet(xs, ys, None, feat_shape=[2, 5])
    self.assert_((d.xs == xs).all())
    ndxs = d.nd_xs()
    self.assert_((ndxs[0,:,:] == np.arange(10).reshape(2, 5)).all())
    self.assert_((ndxs[2,:,:] == np.arange(20, 30).reshape(2, 5)).all())
    ndxs[0, 0, 0] = 1000
    self.assert_(d.xs[0, 0] == 1000)


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
      None, feat_lab=d1.feat_lab, cl_lab=d1.cl_lab)
    d3 = d1 + d2
    self.assert_(d1.nfeatures == d2.nfeatures)
    self.assert_(d1.nclasses == d2.nclasses)
    self.assert_((np.vstack([d1.xs, d2.xs]) == d3.xs).all())
    self.assert_((np.vstack([d1.ys, d2.ys]) == d3.ys).all())

    d4 = DataSet(np.array([[3, 3], [4, 4]]), np.array([[0, 1], [1, 0]]), None)
    self.assert_(d1.nfeatures <> d4.nfeatures)
    self.assertRaises(ValueError, DataSet.__add__, d1, d4)
    
    d5 = DataSet(np.array([[3, 3, 3], [4, 4, 4]]), 
      np.array([[0, 1, 0], [1, 0, 0]]), None)
    
    self.assert_(d1.nclasses <> d5.nclasses)
    self.assertRaises(ValueError, DataSet.__add__, d1, d5)
