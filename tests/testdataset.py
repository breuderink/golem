import unittest
import operator
import numpy as np

from golem import DataSet

class TestDataSetConstruction(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction(self):
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.ninstances_per_class, [1, 1])
    self.assert_((d.nd_xs == d.xs).all())
    self.assertEqual(d.nfeatures, 3)
    self.assertEqual(d.nclasses, 2)
    self.assert_((d.xs == xs).all())
    self.assert_((d.ys == ys).all())
    
  def test_construction_types(self):
    xs = np.arange(12).reshape(-1, 1)
    ys = np.arange(12).reshape(-1, 1)
    ids = np.arange(12).reshape(-1, 1)

    d = DataSet(xs, ys, ids)

    # raise with wrong types
    self.assertRaises(ValueError, DataSet, xs.tolist(), ys, ids)
    self.assertRaises(ValueError, DataSet, xs, ys.tolist(), ids)
    self.assertRaises(ValueError, DataSet, xs, ys, ids.tolist())
    
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, cl_lab = 'c0')
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, feat_lab = 'f0')
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, feat_shape = (1, 1))
    
  def test_construction_dims(self):
    xs = np.arange(12).reshape(-1, 1)
    ys = np.arange(12).reshape(-1, 1)
    ids = np.arange(12).reshape(-1, 1)

    d = DataSet(xs, ys)
    d = DataSet(xs, ys, ids)

    # raise if #rows does not match
    self.assertRaises(ValueError, DataSet, xs[:-1,:], ys, ids);
    self.assertRaises(ValueError, DataSet, xs, ys[:-1, :], ids);
    self.assertRaises(ValueError, DataSet, xs, ys, ids[:-1, :]);
    
    # raise if .ndim <> 2
    self.assertRaises(ValueError, DataSet, xs.flatten(), ys, ids);
    self.assertRaises(ValueError, DataSet, xs, ys.flatten(), ids);
    self.assertRaises(ValueError, DataSet, xs, ys, ids.flatten());
  
  def test_defaults(self):
    xs = np.arange(12).reshape(-1, 1)
    ys = np.arange(12).reshape(-1, 1)
    ids = np.arange(12).reshape(-1, 1)

    self.assertRaises(ValueError, DataSet) # no xs, no ys
    self.assertRaises(ValueError, DataSet, xs=xs) # no ys
    self.assertRaises(ValueError, DataSet, ys=ys) # no xs
    
    dxy = DataSet(xs, ys)
    self.assert_((dxy.ids == np.arange(dxy.ninstances).reshape(-1, 1)).all())
    self.assert_(dxy.cl_lab == ['class0'])
    self.assert_(dxy.feat_lab == ['feat0'])
    self.assert_(dxy.feat_shape == [1])
    
  def test_from_default(self):
    xs = np.arange(12).reshape(-1, 1)
    ys = np.arange(12).reshape(-1, 1)
    ids = np.arange(12).reshape(-1, 1)

    d = DataSet(xs, ys, ids, cl_lab= ['c1'], feat_lab=['f1'], 
      feat_shape=[1, 11])

    # test xs
    d2 = DataSet(xs=np.zeros(xs.shape), default=d)
    self.assert_(not (d.xs == d2.xs).all())
    d2 = DataSet(xs=None, default=d)
    self.assert_((d.xs == d2.xs).all())
    
    # test ys
    d2 = DataSet(ys=np.zeros(ys.shape), default=d)
    self.assert_(not (d.ys == d2.ys).all())
    d2 = DataSet(ys=None, default=d)
    self.assert_((d.ys == d2.ys).all())
    
    # test ids
    d2 = DataSet(ids=ids+1, default=d)
    self.assert_(not (d.ids == d2.ids).all())
    d2 = DataSet(ids=None, default=d)
    self.assert_((d.ids == d2.ids).all())

    # test cl_lab
    d2 = DataSet(cl_lab=['altc0'], default=d)
    self.assert_(d.cl_lab <> d2.cl_lab)
    d2 = DataSet(cl_lab=None, default=d)
    self.assert_(d.cl_lab == d2.cl_lab)

    # test feat_lab
    d2 = DataSet(feat_lab=['altf0'], default=d)
    self.assert_(d.feat_lab <> d2.feat_lab)
    d2 = DataSet(feat_lab=None, default=d)
    self.assert_(d.feat_lab == d2.feat_lab)
    
    # test feat_shape
    d2 = DataSet(feat_shape=[1, 1], default=d)
    self.assert_(d.feat_shape <> d2.feat_shape)
    d2 = DataSet(feat_shape=None, default=d)
    self.assert_(d.feat_shape == d2.feat_shape)
  
  def test_integrity(self):
    xs = np.arange(12).reshape(-1, 1)
    ys = np.arange(12).reshape(-1, 1)
    ids = np.arange(12).reshape(-1, 1)

    self.assertRaises(AssertionError, DataSet, xs, ys, np.zeros(ids.shape))
    self.assertRaises(ValueError, DataSet, xs, ys, feat_lab=['f0', 'f1'])
    self.assertRaises(ValueError, DataSet, xs, ys, cl_lab=['c0', 'c1'])


class TestDataSet(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    ids = np.array([[3], [4]])
    self.d = DataSet(xs, ys, ids, feat_lab=['f1', 'f2', 'f3'], 
      cl_lab=['A', 'B'], feat_shape=[3, 1])
      
  def test_equality(self):
    d = self.d
    self.assert_(d == d)
    self.assert_(d == DataSet(d.xs, d.ys, d.ids, feat_lab=d.feat_lab, 
      cl_lab=d.cl_lab, feat_shape=d.feat_shape))

    # test all kinds of differences
    self.assert_(d <> DataSet(xs=d.xs+1, default=d))
    self.assert_(d <> DataSet(ys=d.ys+1, default=d))
    self.assert_(d <> DataSet(ids=d.ids+1, default=d))
    self.assert_(d <> DataSet(cl_lab=['a', 'b'], default=d))
    self.assert_(d <> DataSet(feat_lab=['F1', 'F2', 'F3'], default=d))
    self.assert_(d <> DataSet(feat_shape=[1, 3], default=d))
    
    # test special cases
    self.assert_(d == DataSet(d.xs.copy(), d.ys.copy(), d.ids.copy(), 
      default=d))
    self.assert_(d <> 3)

  def test_indexing(self):
    '''Test the indexing of DataSet.'''
    d = self.d
    self.assert_(d[:] == d)
    self.assert_(d[0] + d[1] == d)
    self.assert_(d[:-1] == d[0])
    self.assert_(d[0].ninstances == 1)
    self.assert_(d[0].nclasses == d.nclasses)
    self.assert_(d[0].nfeatures == d.nfeatures)
    self.assert_(d[-1] == d[d.nclasses - 1])

    indices = np.arange(d.ninstances)
    self.assert_(d[indices==1] == d[1])
    self.assert_(d[indices.tolist()] == d)
    self.assert_(d[[1]] == d[1])

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
    self.assert_((d.nd_xs[0,:,:] == np.arange(10).reshape(2, 5)).all())
    self.assert_((d.nd_xs[2,:,:] == np.arange(20, 30).reshape(2, 5)).all())
    d.nd_xs[0, 0, 0] = 1000
    self.assert_(d.xs[0, 0] == 1000)
  
  def test_sorted(self):
    '''Test sorting the DataSet'''
    ids = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]]).T
    xs, ys = np.random.random((6, 2)), np.ones((6, 1)) 
    d1d = DataSet(xs, ys, ids[:, 0].reshape(-1, 1))
    d2d = DataSet(xs, ys, ids)

    # shuffle and sort
    shuf_i = np.arange(d1d.ninstances)
    np.random.shuffle(shuf_i)
    d1ds, d2ds = d1d[shuf_i], d2d[shuf_i]
    self.failIf(d1d == d1ds)
    self.failIf(d2d == d2ds)
    d1ds = d1ds.sorted()
    d2ds = d2ds.sorted()
    
    # Test if ids are sorted
    self.assert_(d1d == d1ds)
    self.assert_(d2d == d2ds)

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
