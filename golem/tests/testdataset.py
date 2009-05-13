import unittest
import operator
import tempfile
import numpy as np

from .. import DataSet

class TestDataSetConstruction(unittest.TestCase):
  def test_construction(self):
    '''Test basic construction of DataSet'''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    d = DataSet(xs=xs, ys=ys)
    np.testing.assert_equal(d.xs, xs)
    np.testing.assert_equal(d.ys, ys)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.ninstances_per_class, [1, 1])
    self.assertEqual(d.nfeatures, 3)
    self.assertEqual(d.nclasses, 2)
    np.testing.assert_equal(d.nd_xs, d.xs)
    np.testing.assert_equal(d.ids, np.arange(d.ninstances).reshape(-1, 1))
    self.assertEqual(d.feat_dim_lab, ['feat_dim0'])
    self.assertEqual(d.feat_nd_lab, None)
    self.assertEqual(d.cl_lab, ['class0', 'class1'])
    self.assertEqual(d.feat_lab, None)
    self.assertEqual(d.feat_shape, (3,))
    self.assertEqual(d.extra, {})

  def test_construction_empty(self):
    '''Test empty construction of DataSet'''
    xs = np.zeros((0, 0))
    ys = np.zeros((0, 0))
    d = DataSet(xs, ys)
    self.assertEqual(d.ninstances, 0)
    self.assertEqual(d.ninstances_per_class, [])
    self.assertEqual(d.nfeatures, 0)
    self.assertEqual(d.nclasses, 0)
    self.assertEqual(d.extra, {})

    self.assertRaises(ValueError, DataSet) # no xs, no ys
    self.assertRaises(ValueError, DataSet, xs=xs) # no ys
    self.assertRaises(ValueError, DataSet, ys=ys) # no xs
    
  def test_construction_types(self):
    '''Test types of DataSet during construction'''
    xs = ys = ids = np.arange(12).reshape(-1, 1)
    d = DataSet(xs, ys, ids)

    # raise with wrong types
    self.assertRaises(ValueError, DataSet, xs.tolist(), ys, ids)
    self.assertRaises(ValueError, DataSet, xs, ys.tolist(), ids)
    self.assertRaises(ValueError, DataSet, xs, ys, ids.tolist())
    
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, cl_lab='c0')
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, feat_lab='f0')
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, feat_shape=[1, 1])
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, extra='baz')

    self.assertRaises(AssertionError, DataSet, xs, ys, ids, 
      feat_dim_lab='baz')
    self.assertRaises(AssertionError, DataSet, xs, ys, ids, feat_nd_lab=['a'])
    
  def test_construction_dims(self):
    '''Test the handling of dimension during DataSet construction'''
    xs = ys = ids = np.arange(12).reshape(-1, 1)

    # raise if #rows does not match
    self.assertRaises(ValueError, DataSet, xs[:-1,:], ys, ids);
    self.assertRaises(ValueError, DataSet, xs, ys[:-1, :], ids);
    self.assertRaises(ValueError, DataSet, xs, ys, ids[:-1, :]);
    
    # raise if .ndim != 2
    self.assertRaises(ValueError, DataSet, xs.flatten(), ys, ids);
    self.assertRaises(ValueError, DataSet, xs, ys.flatten(), ids);
    self.assertRaises(ValueError, DataSet, xs, ys, ids.flatten());

    # raise if #lab != #feat
    self.assertRaises(ValueError, DataSet, xs, ys, feat_lab=['f0', 'f1'])
    self.assertRaises(ValueError, DataSet, xs, ys, cl_lab=['c0', 'c1'])

  def test_construction_ids(self):
    '''Test the uniqueness of ids'''
    xs = ys = ids = np.arange(12).reshape(-1, 1)
    ids[0] = ids[1]
    self.assertRaises(ValueError, DataSet, xs, ys, ids)

  def test_construction_feat_shape(self):
    '''Test feat_shape in construction of DataSet'''
    xs = np.arange(12 * 3).reshape(3, -1)
    ys = np.arange(3).reshape(-1, 1)

    DataSet(xs, ys, feat_shape=(12,))
    DataSet(xs, ys, feat_shape=(1, 12))
    self.assertRaises(ValueError, DataSet, xs, ys, feat_shape=(1, 1))

  def test_construction_feat_dim_lab(self):
    '''Test feat_dim_lab in construction of DataSet'''
    xs = np.arange(12 * 3).reshape(3, -1)
    ys = np.arange(3).reshape(-1, 1)

    DataSet(xs, ys, feat_dim_lab=['sec'])
    DataSet(xs, ys, feat_shape=(1, 12), feat_dim_lab=['sec', 'm'])
    self.assertRaises(ValueError, DataSet, xs, ys, feat_shape=(1, 12), 
      feat_dim_lab=['sec'])

  def test_construction_feat_nd_labs(self):
    '''Test feat_nd_lab in construction of DataSet'''
    xs = np.arange(6 * 3).reshape(3, -1)
    ys = np.arange(3).reshape(-1, 1)

    DataSet(xs, ys, feat_nd_lab=[['a', 'b', 'c', 'd', 'e', 'f']])
    DataSet(xs, ys, feat_shape=(2, 3), 
      feat_nd_lab=[['a', 'b'], ['x', 'y', 'z']])
    self.assertRaises(ValueError, DataSet, xs, ys, feat_shape=(3, 2), 
      feat_nd_lab=[['a', 'b'], ['x', 'y', 'z']])
    
  def test_from_default(self):
    xs = ys = ids = np.arange(12).reshape(-1, 1)
    d = DataSet(xs, ys, ids, cl_lab= ['c1'], feat_lab=['f1'], 
      feat_shape=(1, 1), feat_dim_lab=['sec', 'm'], 
      feat_nd_lab=[['y'], ['x']], extra={'foo':'bar'})

    # test xs
    d2 = DataSet(xs=np.zeros(xs.shape), default=d)
    np.testing.assert_equal(d2.xs, np.zeros(xs.shape))
    d2 = DataSet(xs=None, default=d)
    np.testing.assert_equal(d2.xs, d.xs)
    
    # test ys
    d2 = DataSet(ys=np.zeros(ys.shape), default=d)
    np.testing.assert_equal(d2.ys, np.zeros(ys.shape))
    d2 = DataSet(ys=None, default=d)
    np.testing.assert_equal(d2.ys, d.ys)
    
    # test ids
    d2 = DataSet(ids=ids+1, default=d)
    np.testing.assert_equal(d2.ids, ids+1)
    d2 = DataSet(ids=None, default=d)
    np.testing.assert_equal(d2.ids, d.ids)

    # test cl_lab
    d2 = DataSet(cl_lab=['altc0'], default=d)
    self.assertEqual(d2.cl_lab, ['altc0'])
    d2 = DataSet(cl_lab=None, default=d)
    self.assertEqual(d2.cl_lab, d.cl_lab)

    # test feat_lab
    d2 = DataSet(feat_lab=['altf0'], default=d)
    self.assertEqual(d2.feat_lab, ['altf0'])
    d2 = DataSet(feat_lab=None, default=d)
    self.assertEqual(d2.feat_lab, d.feat_lab)
    d2 = DataSet(xs=np.random.rand(12, 6), feat_shape=(2, 3), default=d)
    self.assertEqual(d2.feat_lab, None)
    
    # test feat_shape
    d2 = DataSet(feat_shape=(1, 1, 1), default=d)
    self.assertEqual(d2.feat_shape, (1, 1, 1))
    d2 = DataSet(feat_shape=None, default=d)
    self.assertEqual(d2.feat_shape, d.feat_shape)

    # test feat_dim_lab
    d2 = DataSet(feat_dim_lab=['m', 's'], default=d)
    self.assertEqual(d2.feat_dim_lab, ['m', 's'])
    d2 = DataSet(feat_dim_lab=None, default=d)
    self.assertEqual(d2.feat_dim_lab, d.feat_dim_lab)

    # test feat_nd_lab
    d2 = DataSet(feat_nd_lab=[['a'], ['b']], default=d)
    self.assertEqual(d2.feat_nd_lab, [['a'], ['b']])
    d2 = DataSet(feat_nd_lab=None, default=d)
    self.assertEqual(d2.feat_nd_lab, d.feat_nd_lab)
    d2 = DataSet(xs=np.random.rand(12, 6), feat_shape=(2, 3), default=d)
    self.assertEqual(d2.feat_nd_lab, None)

    # test extra
    d2 = DataSet(extra={'foo':'baz'}, default=d)
    self.assertEqual(d2.extra, {'foo':'baz'})
    d2 = DataSet(extra=None, default=d)
    self.assertEqual(d2.extra, d.extra)

    # test access of emtpy members
    d2 = DataSet(default=DataSet(xs=d.xs, ys=d.ys))
  

class TestDataSet(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    xs = np.array([[0, 0, 0], [1, 1, 1]])
    ys = np.array([[0, 1], [1, 0]])
    ids = np.array([[3], [4]])
    self.d = d = DataSet(xs, ys, ids, feat_lab=['f1', 'f2', 'f3'], 
      cl_lab=['A', 'B'], feat_shape=(3, 1), feat_dim_lab=['d0', 'd1'], 
      feat_nd_lab=[['f1', 'f2', 'f3'],['n']], extra={'foo':'bar'})

    self.diff_ds = [DataSet(xs=d.xs+1, default=d),
      DataSet(ys=d.ys+1, default=d),
      DataSet(ids=d.ids+1, default=d),
      DataSet(cl_lab=['a', 'b'], default=d),
      DataSet(feat_lab=['F1', 'F2', 'F3'], default=d),
      DataSet(feat_shape=(1, 3), feat_nd_lab=[], default=d),
      DataSet(feat_dim_lab=['da', 'db'], default=d),
      DataSet(feat_nd_lab=[['F1', 'F2', 'F3'],['N']], default=d),
      DataSet(extra={'foo':'baz'}, default=d)]
      
  def test_equality(self):
    d = self.d
    self.assertEqual(d, d)
    self.assertEqual(d, DataSet(d.xs, d.ys, d.ids, feat_lab=d.feat_lab, 
      cl_lab=d.cl_lab, feat_shape=d.feat_shape, 
      feat_dim_lab=d.feat_dim_lab, feat_nd_lab=d.feat_nd_lab, 
      extra=d.extra))

    # test all kinds of differences
    for dd in self.diff_ds:
      self.failIfEqual(dd, d)
    
    # test special cases
    self.assertEqual(d, DataSet(d.xs.copy(), d.ys.copy(), d.ids.copy(), 
      default=d))
    self.failIfEqual(d, 3)

  def test_hash(self):
    d = self.d[::2] # noncontiguous arrays can pose a problem
    self.assertEqual(d.hash(), d.hash())
    self.assertEqual(d.hash(), DataSet(d.xs, d.ys, d.ids, feat_lab=d.feat_lab, 
      cl_lab=d.cl_lab, feat_shape=d.feat_shape, 
      feat_dim_lab=d.feat_dim_lab, feat_nd_lab=d.feat_nd_lab, 
      extra=d.extra).hash())

    # test all kinds of differences
    for dd in self.diff_ds:
      self.failIfEqual(dd.hash(), d.hash())
    
    # test special cases
    self.assertEqual(d.hash(),
      DataSet(d.xs.copy(), d.ys.copy(), d.ids.copy(), default=d).hash())

  def test_add(self):
    '''Test the creation of compound datasets using the add-operator.'''
    ids = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]]).T
    xs, ys = np.random.random((6, 3)), np.ones((6, 3)) 
    d = DataSet(xs, ys, ids, feat_lab=['feat%d' for d in range(3)])

    da, db = d[:3], d[3:]
    self.assertEqual(da + db, d)

    # different nfeatures
    self.assertRaises(ValueError, da.__add__,
      DataSet(xs=db.xs[:,:-1], feat_lab=d.feat_lab[:-1], default=db))
    
    # different nclasses
    self.assertRaises(ValueError, da.__add__,
      DataSet(ys=db.ys[:,:-1], cl_lab=d.cl_lab[:-1], default=db))

    # different feat_lab
    self.assertRaises(ValueError, da.__add__,
      DataSet(feat_lab=['f0', 'f1', 'f2'], default=db))

    # different feat_shape
    self.assertRaises(ValueError, da.__add__,
      DataSet(feat_shape=(3, 1), default=db))
    
    # different cl_lab
    self.assertRaises(ValueError, da.__add__,
      DataSet(cl_lab=['c0', 'c1', 'c2'], default=db))

    # different feat_dim_lab
    self.assertRaises(ValueError, da.__add__,
      DataSet(feat_dim_lab=['cm'], default=db))

    # different feat_nd_lab
    self.assertRaises(ValueError, da.__add__,
      DataSet(feat_nd_lab=[['l0', 'l1', 'l2']], default=db))

    # different extra
    self.assertRaises(ValueError, da.__add__,
      DataSet(extra={'foo':'baz'}, default=db))

  def test_indexing(self):
    '''Test the indexing of DataSet.'''
    d = self.d
    # check if all members are correctly extracted
    d0 = DataSet(xs=d.xs[0,:].reshape(1, -1), ys=d.ys[0,:].reshape(1, -1), 
      ids=d.ids[0,:].reshape(1, -1), default=d)
    self.assertEqual(d[0], d0)

    # test high-level properties
    self.assertEqual(d[:], d)
    self.assertEqual(d[0] + d[1], d)
    self.assertEqual(d[:-1], d[0])

    # test various indexing types
    indices = np.arange(d.ninstances)
    self.assertEqual(d[indices==1], d[1])
    self.assertEqual(d[indices.tolist()], d)
    self.assertEqual(d[[1]], d[1])

  def test_class_extraction(self):
    '''Test the extraction of a single class from DataSet'''
    d = self.d
    dA = d.get_class(0)
    dB = d.get_class(1)
    self.assertEqual(dA, d[1])
    self.assertEqual(dB, d[0])

  def test_nd_xs(self):
    '''Test multidimensional xs'''
    xs = np.arange(100).reshape(10, 10)
    ys = np.ones((10, 1))
    d = DataSet(xs, ys, None, feat_shape=(2, 1, 5))
    np.testing.assert_equal(d.xs, xs)
    self.assertEqual(d.ninstances, 10)
    self.assertEqual(d.nfeatures, 10)
    np.testing.assert_equal(
      d.nd_xs[0,:,:], np.arange(10).reshape(2, 1, 5))
    np.testing.assert_equal(
      d.nd_xs[2,:,:], np.arange(20, 30).reshape(2, 1, 5))

    # test modification
    d.nd_xs[0, 0, 0] = 1000
    self.assertEqual(d.xs[0, 0], 1000)
  
  def test_shuffle(self):
    '''Test shuffling the DataSet'''
    xs, ys = np.random.random((6, 2)), np.ones((6, 1)) 
    d = DataSet(xs, ys)

    # shuffle and sort
    ds = d.shuffled()
    self.failIfEqual(ds, d)
    self.assertEqual(ds[np.argsort(ds.ids.flat)], d)

  def test_sorted(self):
    '''Test sorting the DataSet'''
    ids = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]]).T
    xs, ys = np.random.random((6, 2)), np.ones((6, 1)) 
    d1d = DataSet(xs, ys, ids[:, 0].reshape(-1, 1))
    d2d = DataSet(xs, ys, ids)

    # shuffle and sort
    d1ds = d1d.shuffled()
    self.failIfEqual(d1d, d1ds)
    self.assertEqual(d1d, d1ds.sorted())

    d2ds = d2d.shuffled()
    self.failIfEqual(d2d, d2ds)
    self.assertEqual(d2d, d2ds.sorted())

  def test_str(self):
    '''Test string representation.'''
    self.assertEqual("DataSet with 2 instances, 3 features, 2 classes: [1, 1]"
      ", extras: ['foo']", str(self.d))

  def test_save_load(self):
    '''Test loading and saving datasets'''
    # test round-trip using file objects
    _, tfname = tempfile.mkstemp('.goldat')
    self.d.save(open(tfname, 'wb'))
    self.assertEqual(self.d, DataSet.load(open(tfname, 'rb')))

    # test round-trip using filenames
    _, tfname = tempfile.mkstemp('.goldat')
    self.d.save(tfname)
    self.assertEqual(self.d, DataSet.load(tfname))
