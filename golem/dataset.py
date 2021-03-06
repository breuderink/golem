import itertools, warnings
import cPickle
from hashlib import sha1
import numpy as np
import helpers

class DataSet:
  """
  A data set consists of samples with features and labels, and some additional descriptors.
  
  xs*  Samples with features. Features can be multi-dimensional. For example, in the case 
       of EEG, samples can be epochs, consisting of time samples for each channel.
       If xs is multi-dimensional, then the multi-dimensional version can be obtained
       through the property: nd_xs.
  ys*  The true class labels (or values) for the samples. The ground truth. 
       For each sample, for each class, an indication is given as to the truth or chance 
       or value for this class. For example, if for a certain sample the labels are 
       [0 1 0] this means this sample belongs to the second class. The names of the classes 
       are stored in cl_lab.
  ids  A unique identifier per sample.
       If not provided, it will generate a unique integer id from 0 to the number of 
       samples.
       In the case of EEG, this can contain the time stamps of the samples       
  cl_lab A list of string descriptors for each class.
  feat_lab A list of string descriptors for each feature. 
  feat_shape If the features of xs are multi-dimensional, feat_shape contains the shape of
       these features. 
  feat_dim_lab For each feature dimension, a string descriptor.
       In the case of EEG, this could be ['channels', 'time'].
  feat_nd_lab For each feature dimension, a list of string feature descriptors.
       In the case of EEG, this could be [['C3','C4'],['0.01','0.02','0.03','0.04']]
  extra A dictionary that can be used to store any additional information you may want to
       include.
  default A default dataset from which all the information will be obtained that is not
       defined in the initialization of a new dataset.
  Fields with an asterisk have to be provided when creating a new dataset.
  
  For security, it is not possible to write to an already created dataset (xs, ys, and ids
  are locked). This way, you can be certain that a dataset will not be modified from 
  analysis chain to another.
  
  Handy class functions:
  nd_xs          Return a multi-dimensional view of xs, depends on feat_shape.
  save(filename) Store the dataset to disk.
  load(filename) Load a dataset from disk.
  d3 = d1 + d2   Adding datasets together.
  if d1 == d2    Comparing datasets.
  len(d)         Return the number of samples in the dataset.
  d[5]           Return the sample with index 5.
  str(d)         Return a string representation of the dataset.
  d.shuffled()   Return a dataset copy with the samples shuffled.
  d.sorted()     Return a dataset copy with the samples sorted according to ids.
  """

  @property
  def xs(self):
    warnings.warn('DataSet.xs is deprecated, use DataSet.X.T instead.', 
      DeprecationWarning)
    return self.X.T

  @property
  def ys(self):
    warnings.warn('DataSet.ys is deprecated, use DataSet.Y.T instead.', 
      DeprecationWarning)
    return self.Y.T

  @property
  def ids(self):
    warnings.warn('DataSet.ids is deprecated, use DataSet.I.T instead', 
      DeprecationWarning)
    return self.I.T

  @property
  def nd_xs(self):
    warnings.warn('DataSet.nd_xs is deprecated. ' + 
      'Use np.rollaxis(d.ndX, -1) instead.', DeprecationWarning)
    return np.rollaxis(self.ndX, -1)

  def __init__(self, xs=None, ys=None, ids=None, cl_lab=None, feat_lab=None, 
    feat_shape=None, feat_dim_lab=None, feat_nd_lab=None, extra=None, 
    default=None, X=None, Y=None, I=None):
    '''
    Create a new dataset.
    '''
    # backwards compatibility
    if xs != None:
      X = np.asarray(xs).T
    if ys != None:
      Y = np.asarray(ys).T
    if ids != None:
      I = np.asarray(ids).T

    # first, take care of X, Y, I
    if default != None:
      assert isinstance(default, DataSet), 'Default is not a DataSet'
      X = X if X != None else default.X
      Y = Y if Y != None else default.Y
      I = I if I != None else default.I

    if X == None: raise ValueError, 'No X given'
    if Y == None: raise ValueError, 'No Y given'

    # convert to np.ndarray
    self.X, self.Y = X, Y = np.atleast_2d(X, Y)

    if I == None: 
      I = np.arange(self.ninstances)
    self.I = I = np.atleast_2d(I)

    # test essential properties
    if self.X.ndim != 2:
      raise ValueError('Only 2D arrays are supported for X. See feat_shape.')
    if self.Y.ndim != 2:
      raise ValueError('Only 2D arrays are supported for Y.')
    if self.I.ndim != 2:
      raise ValueError('Only 2D arrays are supported for I.')
    if not (self.X.shape[1] == self.Y.shape[1] == self.I.shape[1]):
      raise ValueError('Number of instances (cols) does not match')
    if np.unique(self.I[0]).size != self.ninstances:
      raise ValueError('The ids not unique.')

    if not np.all(np.isfinite(self.X)):
      raise ValueError('Only finite values are allowed for X')
    
    # Lock X, Y, I:
    for arr in [self.X, self.Y, self.I]:
      arr.flags.writeable = False

    # Ok, X, Y and I are ok. Now wel add required structural info:
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
    self.feat_lab = feat_lab
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
    return self[self.Y[i] == 1]

  def sorted(self):
    '''Return a DataSet sorted on the first row of .I'''
    return self[np.argsort(self.I[0])]

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
      return DataSet(X=self.X[:,i], Y=self.Y[:,i], I=self.I[:,i], default=self)
    elif isinstance(i, int):
      return DataSet(X=np.atleast_2d(self.X[:,i]).T,
        Y=np.atleast_2d(self.Y[:,i]).T, I=np.atleast_2d(self.I[:,i]).T,
        default=self)
    else:
      raise ValueError, 'Unknown indexing type.'

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
    if a.X.ndim == 0:
      return b
    if b.X.ndim == 0:
      return a

    # Check for compatibility
    if (a.nfeatures != b.nfeatures) or (a.nclasses != b.nclasses):
      raise ValueError, 'The #features or #classes do not match'
    for member in a.__dict__.keys():
      if member not in ['X', 'Y', 'I']:
        if a.__dict__[member] != b.__dict__[member]:
          raise ValueError('Cannot add DataSets: %s is different' % member)

    return DataSet(X=np.hstack([a.X, b.X]), Y=np.hstack([a.Y, b.Y]),
      I=np.hstack([a.I, b.I]), default=a)

  def __eq__(a, b):
    if not isinstance(b, DataSet):
      return False
    for member in a.__dict__.keys():
      am, bm = a.__dict__[member], b.__dict__[member]
      if isinstance(am, np.ndarray):
        if am.shape != bm.shape or not np.all(am == bm):
          return False
      else:
        if not am == bm:
          return False
    return True
    
  def __ne__(a, b):
    return not a == b
    
  @property
  def nclasses(self):
    if self.Y.ndim == 0:
      return 0
    return self.Y.shape[0]
        
  @property
  def ninstances(self):
    if self.X.ndim == 0:
      return 0
    return self.X.shape[1]
    
  @property
  def ninstances_per_class(self):
    return np.sum(helpers.hard_max(self.Y), axis=1).astype(int).tolist()

  @property
  def prior(self):
    return np.asarray(self.ninstances_per_class) / float(self.ninstances)

  @property
  def nfeatures(self):
    if self.X.ndim == 0:
      return 0
    return self.X.shape[0]

  @property
  def ndX(self):
    '''Return multi-dimensional view of X'''
    return self.X.reshape(self.feat_shape + (self.ninstances,))


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
