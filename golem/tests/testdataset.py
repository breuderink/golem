import unittest
import operator
import tempfile
import numpy as np

from .. import DataSet

class TestDataSetConstruction(unittest.TestCase):
  def test_construction_old(self):
    '''Test basic construction of DataSet'''
    xs = [[0, 0, 0], [1, 1, 1]]
    ys = [[0, 1], [1, 0]]
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

  def test_construction(self):
    '''Test basic construction of DataSet'''
    X = [[0, 1], [0, 1], [0, 1]]
    Y = [[0, 1], [1, 0]]
    d = DataSet(X=X, Y=Y)
    np.testing.assert_equal(d.X, X)
    np.testing.assert_equal(d.xs.T, X)
    np.testing.assert_equal(d.Y, Y)
    np.testing.assert_equal(d.ys.T, Y)
    self.assertEqual(d.ninstances, 2)
    self.assertEqual(d.ninstances_per_class, [1, 1])
    self.assertEqual(d.nfeatures, 3)
    self.assertEqual(d.nclasses, 2)
    np.testing.assert_equal(d.nd_xs, d.xs)
    np.testing.assert_equal(d.I, np.atleast_2d(np.arange(d.ninstances)))
    np.testing.assert_equal(d.ids.T, d.I)
    self.assertEqual(d.feat_dim_lab, ['feat_dim0'])
    self.assertEqual(d.feat_nd_lab, None)
    self.assertEqual(d.cl_lab, ['class0', 'class1'])
    self.assertEqual(d.feat_lab, None)
    self.assertEqual(d.feat_shape, (3,))
    self.assertEqual(d.extra, {})

  def test_construction_empty(self):
    '''Test empty construction of DataSet'''
    X = np.zeros((0, 0))
    Y = np.zeros((1, 0))
    d = DataSet(X=X, Y=Y)
    self.assertEqual(d.ninstances, 0)
    self.assertEqual(d.ninstances_per_class, [0])
    self.assertEqual(d.nfeatures, 0)
    self.assertEqual(d.nclasses, 1)
    self.assertEqual(d.extra, {})

    self.assertRaises(ValueError, DataSet) # no X, no Y
    self.assertRaises(ValueError, DataSet, X=X) # no Y
    self.assertRaises(ValueError, DataSet, Y=Y) # no X
    
  def test_construction_types(self):
    '''Test types of DataSet during construction'''
    X = Y = I = np.arange(12)
    # types for X, Y and I are not tested, as mainly there shape is of
    # importance
    
    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, cl_lab='c0')
    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, feat_lab='f0')
    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, feat_shape=[1, 1])
    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, extra='baz')

    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, 
      feat_dim_lab='baz')
    self.assertRaises(AssertionError, DataSet, X=X, Y=Y, I=I, feat_nd_lab=['a'])
    
  def test_construction_dims(self):
    '''Test the handling of dimensions during DataSet construction'''
    X = Y = I = np.arange(12)

    # raise if #rows does not match
    self.assertRaises(ValueError, DataSet, X=X[:-1], Y=Y, I=I)
    self.assertRaises(ValueError, DataSet, X=X, Y=Y[:-1], I=I)
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, I=I[:-1])
    
    # raise if .ndim >= 2
    self.assertRaises(ValueError, DataSet, X=X.reshape(2, 2, 3), Y=Y[:3], 
      I=I[:3])
    self.assertRaises(ValueError, DataSet, X=X[:3], Y=Y.reshape(2, 2, 3), 
      I=I[:3])
    self.assertRaises(ValueError, DataSet, X=X[:3], Y=Y[:3], 
      I=I.reshape(2, 2, 3))

    # raise if #lab != #feat
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, feat_lab=['f0', 'f1'])
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, cl_lab=['c0', 'c1'])

  def test_construction_ids(self):
    '''Test the uniqueness of I check'''
    X = Y = I = np.arange(12)
    I[0] = I[1]
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, I=I)

  def test_construction_feat_shape(self):
    '''Test feat_shape in construction of DataSet'''
    X = np.arange(12 * 3).reshape(12, 3)
    Y = np.arange(3)

    DataSet(X=X, Y=Y, feat_shape=(12,))
    DataSet(X=X, Y=Y, feat_shape=(1, 12))
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, feat_shape=(1, 11))

  def test_construction_feat_dim_lab(self):
    '''Test feat_dim_lab in construction of DataSet'''
    X = np.arange(12 * 3).reshape(12, 3)
    Y = np.arange(3)

    DataSet(X=X, Y=Y, feat_dim_lab=['sec'])
    DataSet(X=X, Y=Y, feat_shape=(1, 12), feat_dim_lab=['sec', 'm'])
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, feat_shape=(1, 12), 
      feat_dim_lab=['sec'])

  def test_construction_feat_nd_labs(self):
    '''Test feat_nd_lab in construction of DataSet'''
    X = np.arange(6 * 3).reshape(6, 3)
    Y = np.arange(3)

    DataSet(X=X, Y=Y, feat_nd_lab=[['a', 'b', 'c', 'd', 'e', 'f']])
    DataSet(X=X, Y=Y, feat_shape=(2, 3), 
      feat_nd_lab=[['a', 'b'], ['x', 'y', 'z']])
    self.assertRaises(ValueError, DataSet, X=X, Y=Y, feat_shape=(3, 2), 
      feat_nd_lab=[['a', 'b'], ['x', 'y', 'z']])
    
  def test_from_default(self):
    X = Y = I = np.atleast_2d(np.arange(12))
    d = DataSet(X=X, Y=Y, I=I, cl_lab= ['c1'], feat_lab=['f1'], 
      feat_shape=(1, 1), feat_dim_lab=['sec', 'm'], 
      feat_nd_lab=[['y'], ['x']], extra={'foo':'bar'})

    # test X
    d2 = DataSet(X=X+1, default=d)
    np.testing.assert_equal(d2.X, X+1)
    d2 = DataSet(X=None, default=d)
    np.testing.assert_equal(d2.X, d.X)
    
    # test Y
    d2 = DataSet(Y=Y+1, default=d)
    np.testing.assert_equal(d2.Y, Y+1)
    d2 = DataSet(Y=None, default=d)
    np.testing.assert_equal(d2.Y, d.Y)
    
    # test I
    d2 = DataSet(I=I+1, default=d)
    np.testing.assert_equal(d2.I, I+1)
    d2 = DataSet(I=None, default=d)
    np.testing.assert_equal(d2.I, d.I)

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
    d2 = DataSet(X=np.random.rand(6, 12), feat_shape=(2, 3), default=d)
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
    d2 = DataSet(X=np.random.rand(6, 12), feat_shape=(2, 3), default=d)
    self.assertEqual(d2.feat_nd_lab, None)

    # test extra
    d2 = DataSet(extra={'foo':'baz'}, default=d)
    self.assertEqual(d2.extra, {'foo':'baz'})
    d2 = DataSet(extra=None, default=d)
    self.assertEqual(d2.extra, d.extra)

  def test_finite_feats(self):
    Y = np.ones((2, 10))
    for v in [np.inf, -np.inf, np.nan]:
      X = np.zeros((2, 10))
      d = DataSet(X=X.copy(), Y=Y) # no error
      X[1, 5] = v
      self.assertRaises(ValueError, DataSet, X=X, Y=Y)


class TestDataSet(unittest.TestCase):
  def setUp(self):
    '''Setup a default DataSet.'''
    X = np.array([[0, 1], [0, 1], [0, 1]])
    Y = np.array([[0, 1], [1, 0]])
    I = np.array([3, 4])
    self.d = d = DataSet(X=X, Y=Y, I=I, feat_lab=['f1', 'f2', 'f3'], 
      cl_lab=['A', 'B'], feat_shape=(3, 1), feat_dim_lab=['d0', 'd1'], 
      feat_nd_lab=[['f1', 'f2', 'f3'],['n']], extra={'foo':'bar'})

  def test_equality(self):
    d = self.d
    diff_ds = [DataSet(X=d.X+1, default=d),
      DataSet(Y=d.Y+1, default=d),
      DataSet(I=d.I+1, default=d),
      DataSet(cl_lab=['a', 'b'], default=d),
      DataSet(feat_lab=['F1', 'F2', 'F3'], default=d),
      DataSet(feat_shape=(1, 3), feat_nd_lab=[], default=d),
      DataSet(feat_dim_lab=['da', 'db'], default=d),
      DataSet(feat_nd_lab=[['F1', 'F2', 'F3'],['N']], default=d),
      DataSet(extra={'foo':'baz'}, default=d),
      d[:0]]

    self.assertEqual(d, d)
    self.assertEqual(d, DataSet(X=d.X, Y=d.Y, I=d.I, feat_lab=d.feat_lab, 
      cl_lab=d.cl_lab, feat_shape=d.feat_shape, feat_dim_lab=d.feat_dim_lab, 
      feat_nd_lab=d.feat_nd_lab, extra=d.extra))

    # test all kinds of differences
    for dd in diff_ds:
      self.failIfEqual(dd, d)
    
    # test special cases
    self.assertEqual(d, DataSet(X=d.X.copy(), Y=d.Y.copy(), I=d.I.copy(), 
      default=d))
    self.failIfEqual(d, 3)
    self.failIfEqual(d[:0], d) # triggered special cast in np.array comparison.
    self.failIfEqual(d[:0], d[0]) # similar

  def test_add(self):
    '''Test the creation of compound datasets using the add-operator.'''
    I = np.array([[0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 0, 0]])
    X, Y = np.random.random((3, 6)), np.ones((3, 6)) 
    d = DataSet(X=X, Y=Y, I=I, feat_lab=['feat%d' for d in range(3)])

    da, db = d[:3], d[3:]
    self.assertEqual(da + db, d)

    # different nfeatures
    self.assertRaises(ValueError, da.__add__,
      DataSet(X=db.X[:-1], feat_lab=d.feat_lab[:-1], default=db))
    
    # different nclasses
    self.assertRaises(ValueError, da.__add__,
      DataSet(Y=db.Y[:-1], cl_lab=d.cl_lab[:-1], default=db))

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
    d0 = DataSet(X=d.X[:,0].reshape(-1, 1), 
      Y=d.Y[:,0].reshape(-1, 1), 
      I=d.I[:,0].reshape(-1, 1), default=d)
    self.assertEqual(d[0], d0)

    # test high-level properties
    self.assertEqual(d[:], d)
    self.assertEqual(d[0] + d[1], d)
    self.assertEqual(d[:-1], d[0])

    # test various indexing types
    indices = np.arange(d.ninstances)
    self.assertEqual(d[indices==0], d[0])
    self.assertEqual(d[indices.tolist()], d)
    self.assertEqual(d[[1]], d[1])

  def test_class_extraction(self):
    '''Test the extraction of a single class from DataSet'''
    d = self.d
    dA = d.get_class(0)
    dB = d.get_class(1)
    self.assertEqual(dA, d[1])
    self.assertEqual(dB, d[0])

  def test_ndX(self):
    '''Test multi-dimensional X'''
    X = np.arange(100).reshape(10, 10).T
    Y = np.ones(10)
    d = DataSet(X=X, Y=Y, feat_shape=(2, 1, 5))
    np.testing.assert_equal(d.X, X)
    self.assertEqual(d.ninstances, 10)
    self.assertEqual(d.nfeatures, 10)
    print d.ndX.shape
    np.testing.assert_equal(
      d.ndX[:,:,:,0], np.arange(10).reshape(2, 1, 5))
    np.testing.assert_equal(
      d.ndX[:,:,:,2], np.arange(20, 30).reshape(2, 1, 5))

    np.testing.assert_equal(np.rollaxis(d.nd_xs, 0, 4), d.ndX)
  
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
    self.assertEqual(
      "DataSet with 2 instances, 3 features [3x1], 2 classes: [1, 1], " + 
        "extras: ['foo']", str(self.d))

  def test_repr(self):
    '''Test string representation.'''
    self.assertEqual(
      "DataSet with 2 instances, 3 features [3x1], 2 classes: [1, 1], " + 
        "extras: ['foo']", repr(self.d))

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

  def test_write_protected(self):
    d = self.d
    for att in [d.xs, d.ys, d.ids]:
      self.assertRaises(RuntimeError, att.__setitem__, (0, 0), -1)

class TestEmpty(unittest.TestCase):
  def setUp(self):
    self.d0 = DataSet(xs=np.zeros((0, 10)), ys=np.zeros((0, 3)))

  def test_props(self):
    d0 = self.d0
    self.assertEqual(d0.ninstances, 0)
    self.assertEqual(d0.nfeatures, 10)
    self.assertEqual(d0.nclasses, 3)
    self.assertEqual(d0.ninstances_per_class, [0, 0, 0])

  def test_sort(self): 
    d0 = self.d0
    ds = d0.sorted()
    self.assertEqual(ds, d0)
    self.assertNotEqual(id(ds), id(d0))

  def test_shuffle(self): 
    d0 = self.d0
    ds = d0.shuffled()
    self.assertEqual(ds, d0)
    self.assertNotEqual(id(ds), id(d0))

  def test_bounds(self):
    d0 = self.d0
    self.assertRaises(IndexError, d0.__getitem__, 0)
    self.assertRaises(IndexError, d0.__getitem__, 1)
    self.assertEqual(d0[:], d0)
    self.assertEqual(d0[[]], d0)
    self.assertEqual(d0[np.asarray([])], d0)
