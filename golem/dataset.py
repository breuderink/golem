import itertools
import cPickle
from hashlib import sha1
import numpy as np
import helpers

class DataSet:
  def __init__(self, xs=None, ys=None, ids=None, cl_lab=None, feat_lab=None, 
    feat_shape=None, default=None):
    '''Create a new dataset.'''
    # First, take care of xs, ys and ids
    if default == None:
      if xs == None: raise ValueError, 'No xs given'
      if ys == None: raise ValueError, 'No ys given'
      self.xs, self.ys = xs, ys
      self.ids = ids if ids != None else\
        np.arange(self.ninstances).reshape(-1, 1)
    else:
      assert isinstance(default, DataSet), 'Default is not a DataSet'
      self.xs = xs if xs != None else default.xs
      self.ys = ys if ys != None else default.ys
      self.ids = ids if ids != None else default.ids

    if not isinstance(self.xs, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for xs'
    if self.xs.ndim != 2:
      raise ValueError, 'Only 2d arrays are supported for xs. See feat_shape.'
    if not isinstance(self.ys, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for ys'
    if self.ys.ndim != 2:
      raise ValueError, 'Only 2d arrays are supported for ys.'
    if not isinstance(self.ids, np.ndarray):
      raise ValueError, 'Only np.ndarray is supported for ids'
    if self.ids.ndim != 2:
      raise ValueError, 'Only 2d arrays are supported for ids.'
    if not (self.xs.shape[0] == self.ys.shape[0] == self.ids.shape[0]):
      raise ValueError, 'Number of rows does not match'
    assert np.unique(self.ids[:,0]).size == self.ninstances, \
      'The ids not unique.'

    # Ok, xs, ys, and ids are ok. Now the labels and shapes
    if default == None:  
      self.cl_lab = cl_lab if cl_lab != None \
        else ['class%d' % i for i in range(self.nclasses)]
      self.feat_lab = feat_lab
      self.feat_shape = feat_shape if feat_shape != None \
        else [self.nfeatures]
    if default != None:
      self.cl_lab = cl_lab if cl_lab != None else default.cl_lab
      self.feat_lab = feat_lab if feat_lab != None else default.feat_lab
      if feat_shape == None:
        if np.prod(default.feat_shape) != self.nfeatures:
          self.feat_shape = [self.nfeatures]
        else:
          self.feat_shape = default.feat_shape
      else:
        self.feat_shape = feat_shape
   
    assert isinstance(self.cl_lab, list), 'Class labels not a list'
    assert self.feat_lab == None or isinstance(self.feat_lab, list), \
      'Feature labels not a list'
    assert isinstance(self.feat_shape, list), 'Feature shape not a list'

    if self.feat_lab != None and len(self.feat_lab) != self.nfeatures:
      raise ValueError, '"%s" does not match #features' % self.feat_lab
    if len(self.cl_lab) != self.nclasses:
      raise ValueError, 'The number of class labels does not match #classes'
    if not np.prod(self.feat_shape) == self.nfeatures:
      raise ValueError, '%d features does not match feat_shape %s' % \
        (self.nfeatures, self.feat_shape)

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
    if (a.nfeatures != b.nfeatures) or (a.nclasses != b.nclasses):
      raise ValueError, 'The #features or #classes do not match'
    if a.feat_lab != b.feat_lab:
      raise ValueError, 'The feature labels do not match'
    if a.feat_shape != b.feat_shape:
      raise ValueError, 'The feature shapes do not match'
    if a.cl_lab != b.cl_lab:
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

  def hash(self):
    '''
    Return a sha1 hash for caching. Does not return a integer as required
    by dictionaries and sets (and is therefore not named __hash__).
    '''
    hash = sha1()
    hash.update(self.xs.view(np.uint8))
    hash.update(self.ys.view(np.uint8))
    hash.update(self.ids.view(np.uint8))
    hash.update(cPickle.dumps((self.feat_lab, self.cl_lab, self.feat_shape)))
    return hash.digest()
    
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
    if self.ninstances == 0:
      return []
    return np.sum(helpers.hard_max(self.ys), axis=0).astype(int).tolist()

  @property
  def nfeatures(self):
    if self.xs.ndim == 0:
      return 0
    return self.xs.shape[1]
  
  @property
  def nd_xs(self):
    '''Return N-dimensional view of xs'''
    if self.feat_shape != None:
      return self.xs.reshape([self.ninstances] + self.feat_shape)
    raise Exception, 'Feature shape is unknown'
    return self.xs.shape[1]

  def save(self, file):
    f = open(file, 'wb') if isinstance(file, str) else file
    cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
    if isinstance(file, str):
      f.close()

  @classmethod
  def load(cls, file):
    f = open(file, 'rb') if isinstance(file, str) else file
    d = cPickle.load(f)
    assert isinstance(d, DataSet)
    if isinstance(file, str):
      f.close()
    return d
