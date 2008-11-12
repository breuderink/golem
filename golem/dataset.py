import itertools
import numpy as np
import helpers

class DataSet:
  def __init__(self, xs=None, ys=None, ids=None, cl_lab=None, feat_lab=None, 
    feat_shape=None, default=None):
    '''Create a new dataset.'''
    if default <> None:
      # Fill in blanks from default DataSet
      assert(isinstance(default, DataSet))
      xs = xs if xs <> None else default.xs
      ys = ys if ys <> None else default.ys
      ids = ids if ids <> None else default.ids
      cl_lab = cl_lab if cl_lab <> None else default.cl_lab
      feat_lab = feat_lab if feat_lab <> None else default.feat_lab
      feat_shape = feat_shape if feat_shape <> None else default.feat_shape
      
    if not isinstance(xs, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for xs'
    if xs.ndim <> 2:
      raise ValueError, 'Only 2d arrays are supported for xs. See feat_shape.'
    if not isinstance(ys, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for ys'
    if ys.ndim <> 2:
      raise ValueError, 'Only 2d arrays are supported for ys.'

    self.xs = xs
    self.ys = ys
    
    self.ids = ids if ids <> None else \
      np.arange(self.ninstances).reshape(-1, 1)
    self.cl_lab = cl_lab if cl_lab <> None \
      else ['class%d' % i for i in range(self.nclasses)]
    self.feat_lab = feat_lab if feat_lab <> None \
      else ['feat%d' % i for i in range(self.nfeatures)]
    self.feat_shape = feat_shape if feat_shape <> None \
      else [self.nfeatures]

    del xs, ys, ids, cl_lab, feat_lab, feat_shape
      
    if not isinstance(self.ids, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for ids'
    if self.ids.ndim <> 2:
      raise ValueError, 'Only 2d arrays are supported for ids.'
    if not (self.xs.shape[0] == self.ys.shape[0] == self.ids.shape[0]):
      raise ValueError, 'Number of rows does not match'
    
    assert(isinstance(self.cl_lab, list))
    assert(isinstance(self.feat_lab, list))
    assert(isinstance(self.feat_shape, list))

    # Final integrity test
    assert(np.unique(self.ids[:,0]).size == self.ninstances)
    if len(self.feat_lab) <> self.nfeatures:
      raise ValueError, '"%s" does not match #features' % self.feat_lab
    if len(self.cl_lab) <> self.nclasses:
      raise ValueError, 'The number of class labels does not match #classes'

  def get_class(self, i):
    return self[self.ys[:, i] == 1]

  def sorted(self):
    '''Sort by id'''
    return self[np.argsort(self.ids[:,0])] # sort on first col of .ids
    
  def __getitem__(self, i):
    if isinstance(i, slice) or isinstance(i, list) or isinstance(i, np.ndarray):
      return DataSet(xs=self.xs[i, :], ys=self.ys[i,:], ids=self.ids[i, :], 
        default=self)
    elif isinstance(i, int):
      return DataSet(xs=self.xs[i, :].reshape(1, -1), 
        ys=self.ys[i,:].reshape(1, -1), ids=self.ids[i,:].reshape(1, -1),
        default=self)
    else:
      raise ValueError, 'Unkown indexing type.'

  def __len__(self):
    return self.ninstances
 
  def __iter__(self):
    for i in range(self.ninstances):
      yield (self.xs[i], self.ys[i])

  def __str__(self):
    return 'DataSet with %d instances, %d features, %d classes: %s' % \
      (self.ninstances, self.nfeatures, self.nclasses, 
      repr(self.ninstances_per_class))

  def __add__(a, b):
    '''Create a new DataSet by adding the instances of b to a'''
    assert(isinstance(a, DataSet))
    assert(isinstance(b, DataSet))

    # Handle empty datasets
    if a.xs.ndim == 0:
      return b
    if b.xs.ndim == 0:
      return a

    # Check shape and labels
    if (a.nfeatures <> b.nfeatures) or (a.nclasses <> b.nclasses):
      raise ValueError, 'The #features or #classes do not match'
    if a.feat_lab <> b.feat_lab:
      raise ValueError, 'The feature labels do not match'
    if a.cl_lab <> b.cl_lab:
      raise ValueError, 'The class labels do not match'

    return DataSet(np.vstack([a.xs, b.xs]), np.vstack([a.ys, b.ys]),
      ids=np.vstack([a.ids, b.ids]), default=a)

  def __eq__(a, b):
    if isinstance(b, DataSet):
      return (a.xs == b.xs).all() and (a.ys == b.ys).all() and \
        (a.ids == b.ids).all() and a.feat_lab == b.feat_lab and \
        a.cl_lab == b.cl_lab and a.feat_shape == b.feat_shape

    return False
    
  def __ne__(a, b):
    return not a == b
    
  @property
  def nclasses(self):
    if self.ys.ndim == 0:
      return 0
    return self.ys.shape[1]
        
  @property
  def ninstances(self):
    if self.xs.ndim == 0:
      return 0
    return self.xs.shape[0]
    
  @property
  def ninstances_per_class(self):
    return np.sum(helpers.hard_max(self.ys), axis=0).astype(int).tolist()

  @property
  def nfeatures(self):
    if self.xs.ndim == 0:
      return 0
    return self.xs.shape[1]
  
  @property
  def nd_xs(self):
    '''Return N-dimensional view of xs'''
    if self.feat_shape <> None:
      return self.xs.reshape([self.ninstances] + self.feat_shape)
    raise Exception, 'Feature shape is unknown'
    return self.xs.shape[1]
