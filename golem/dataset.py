import itertools
import cPickle
from hashlib import sha1
import numpy as np
import helpers

class DataSet:
  def __init__(self, xs=None, ys=None, ids=None, cl_lab=None, feat_lab=None, 
    feat_shape=None, feat_dim_lab=None, feat_nd_lab=None, extra=None, 
    default=None):
    '''
    Create a new dataset.
    '''
    # first, take care of xs, ys, ids
    if default == None:
      if xs == None: raise ValueError, 'No xs given'
      if ys == None: raise ValueError, 'No ys given'
      self.xs, self.ys, self.ids = xs, ys, ids
    else:
      assert isinstance(default, DataSet), 'Default is not a DataSet'
      self.xs = xs if xs != None else default.xs
      self.ys = ys if ys != None else default.ys
      self.ids = ids if ids != None else default.ids

    # convert to np.ndarray
    self.xs = np.asarray(self.xs)
    self.ys = np.asarray(self.ys)

    if self.ids == None: 
      self.ids = np.arange(self.ninstances).reshape(-1, 1)
    self.ids = np.array(self.ids)

    # test essential properties
    if self.xs.ndim != 2:
      raise ValueError('Only 2d arrays are supported for xs. See feat_shape.')
    if self.ys.ndim != 2:
      raise ValueError('Only 2d arrays are supported for ys.')
    if self.ids.ndim != 2:
      raise ValueError('Only 2d arrays are supported for ids.')
    if not (self.xs.shape[0] == self.ys.shape[0] == self.ids.shape[0]):
      raise ValueError('Number of rows does not match')
    if np.unique(self.ids[:,0]).size != self.ninstances:
      raise ValueError('The ids not unique.')

    if np.any(np.isnan(self.xs)):
      raise ValueError('NaN not allowed for xs.')
    if np.any(np.isinf(self.xs)):
      raise ValueError('Infinite values not allowed for xs.')
    
    # Lock xs, ys, ids:
    for arr in [self.xs, self.ys, self.ids]:
      arr.flags.writeable = False

    # Ok, xs, ys, and ids are ok. Now wel add required structural info
    if default != None:  
      # fill in from default arg
      if cl_lab == None: cl_lab = default.cl_lab
      if feat_lab == None: 
        feat_lab = default.feat_lab
        if feat_lab != None and len(feat_lab) != self.nfeatures:
          feat_lab = None
      if feat_shape == None:
        feat_shape = default.feat_shape
        if np.prod(default.feat_shape) != self.nfeatures:
          feat_shape = (self.nfeatures,)

    self.cl_lab = cl_lab if cl_lab \
      else ['class%d' % i for i in range(self.nclasses)]
    self.feat_lab = feat_lab if feat_lab else None
    self.feat_shape = feat_shape if feat_shape != None else (self.nfeatures,)

    # Now we are basically done, but let's add optional info
    if default != None:  
      # fill in from default arg
      if feat_dim_lab == None:
        feat_dim_lab = default.feat_dim_lab
        if len(feat_dim_lab) != len(self.feat_shape):
          feat_dim_lab = None
      if feat_nd_lab == None:
        feat_nd_lab = default.feat_nd_lab
        if feat_nd_lab != None:
          if tuple(len(dim_lab) for dim_lab in feat_nd_lab) != self.feat_shape:
            feat_nd_lab = None
      extra = extra if extra != None else default.extra

    self.feat_dim_lab = feat_dim_lab if feat_dim_lab else \
      ['feat_dim%d' % i for i in range(len(self.feat_shape))]
    self.feat_nd_lab = feat_nd_lab if feat_nd_lab else None
    self.extra = extra if extra else {}

    self.check_consistency()

  def check_consistency(self):
    assert isinstance(self.cl_lab, list), 'cl_lab not a list'
    assert self.feat_lab == None or isinstance(self.feat_lab, list), \
      'Feature labels not a list'
    assert isinstance(self.feat_shape, tuple), 'feat_shape not a tuple'
    assert isinstance(self.feat_dim_lab, list), 'feat_dim_lab not a list'
    assert isinstance(self.extra, dict), 'extra not a dict'

    if self.feat_lab != None and len(self.feat_lab) != self.nfeatures:
      raise ValueError('feat_lab %s does not match #features' % self.feat_lab)
    if len(self.cl_lab) != self.nclasses:
      raise ValueError('The number of class labels does not match #classes')
    if not np.prod(self.feat_shape) == self.nfeatures:
      raise ValueError('%d features does not match feat_shape %s' % \
        (self.nfeatures, self.feat_shape))

    if self.feat_dim_lab != None:
      if len(self.feat_shape) != len(self.feat_dim_lab):
        raise ValueError('feat_dim_lab %s does not match feat_shape %s' %
          (repr(self.feat_dim_lab), repr(self.feat_shape)))
    if self.feat_nd_lab != None:
      assert len(self.feat_nd_lab) == len(self.feat_shape)
      for i, dlab in enumerate(self.feat_nd_lab):
        assert isinstance(dlab, list)
        if len(dlab) != self.feat_shape[i]:
          raise ValueError(
            'feat_nd_lab[%d] %s does not match feat_shape %s' % \
            (i, dlab, self.feat_shape))

  def get_class(self, i):
    return self[self.ys[:, i] == 1]

  def sorted(self):
    '''Return a by ids sorted DataSet'''
    return self[np.argsort(self.ids[:,0])] # sort on first col of .ids

  def shuffled(self):
    '''Return a shuffled DataSet'''
    si = np.arange(self.ninstances)
    np.random.shuffle(si)
    return self[si]
    
  def __getitem__(self, i):
    if isinstance(i, slice) or isinstance(i, list) or isinstance(i, np.ndarray):
      if self.ninstances == 0:
        if not isinstance(i, slice) and len(i) == 0:
          # Because np.zeros((0, 10)[[]] raises error, we use a workaround 
          # using slice to index in a empty dataset.
          # see http://projects.scipy.org/numpy/ticket/1171
          i = slice(0) 
      return DataSet(xs=self.xs[i], ys=self.ys[i], ids=self.ids[i], 
        default=self)
    elif isinstance(i, int):
      return DataSet(xs=self.xs[i].reshape(1, -1), 
        ys=self.ys[i].reshape(1, -1), ids=self.ids[i].reshape(1, -1),
        default=self)
    else:
      raise ValueError, 'Unkown indexing type.'

  def __len__(self):
    return self.ninstances

  def __str__(self):
    return ('DataSet with %d instances, %d features [%s], %d classes: %s, '
      'extras: %s') % (self.ninstances, self.nfeatures, 
      'x'.join([str(di) for di in self.feat_shape]), 
      self.nclasses, repr(self.ninstances_per_class), repr(self.extra.keys()))

  def __repr__(self):
    return str(self)

  def __add__(a, b):
    '''Create a new DataSet by adding the instances of b to a'''
    assert(isinstance(a, DataSet))
    assert(isinstance(b, DataSet))

    # Handle empty datasets
    if a.xs.ndim == 0:
      return b
    if b.xs.ndim == 0:
      return a

    # Check for compatibility
    if (a.nfeatures != b.nfeatures) or (a.nclasses != b.nclasses):
      raise ValueError, 'The #features or #classes do not match'
    for member in a.__dict__.keys():
      if member not in ['xs', 'ys', 'ids']:
        if a.__dict__[member] != b.__dict__[member]:
          raise ValueError('Cannot add DataSets: %s is different' % member)

    return DataSet(np.vstack([a.xs, b.xs]), np.vstack([a.ys, b.ys]),
      ids=np.vstack([a.ids, b.ids]), default=a)

  def __eq__(a, b):
    if not isinstance(b, DataSet):
      return False
    for member in a.__dict__.keys():
      am, bm = a.__dict__[member], b.__dict__[member]
      cmp = am == bm
      if isinstance(cmp, np.ndarray):
        if not cmp.all():  
          return False
      elif not cmp:
        return False
    return True
    
  def __ne__(a, b):
    return not a == b

  def hash(self):
    '''
    Return a sha1 hash for caching. Does not return a integer as required
    by dictionaries and sets (and is therefore not named __hash__).
    '''
    hash = sha1()
    for member in sorted(self.__dict__.keys()):
      v = self.__dict__[member]
      if member in ['xs', 'ys', 'ids']:
        hash.update(np.ascontiguousarray(v).view(np.uint8))
      else:
        hash.update(repr(v))
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
      return self.xs.reshape((self.ninstances,) + self.feat_shape)
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
    d.check_consistency()
    if isinstance(file, str):
      f.close()
    return d
