import itertools
import numpy as np

# - add support for ids (for time etc)

class DataSet:
  def __init__(self, xs=None, ys=None, feature_labels=None, class_labels=None):
    '''Create a new dataset.'''
    if xs == None:
      xs = np.array(None)
      ys = np.array(None)
    elif not (isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)):
      raise ValueError, 'Only numpy.ndarray is supported for xs and ys.'
    elif xs.ndim <> 2 or ys.ndim <> 2:
      raise ValueError, 'Arguments xs and ys should be *2D* numpy arrays'
    elif xs.shape[0] <> ys.shape[0]:
      raise ValueError, 'The #instances does not match #labels'
   
    self.xs = xs
    self.ys = ys

    self.feature_labels = feature_labels if feature_labels else \
      ['feat%d' % i for i in range(self.nfeatures)]
    self.class_labels = class_labels if class_labels else \
      ['class%d' % i for i in range(self.nclasses)]
    
    if len(self.feature_labels) <> self.nfeatures:
      raise ValueError, 'The number of feature labels does not match #features'
    if len(self.class_labels) <> self.nclasses:
      raise ValueError, 'The number of class labels does not match #classes'

  def get_class(self, i):
    xs = self.xs[self.ys[:, i] == 1]
    ys = self.ys[self.ys[:, i] == 1]
    if xs.ndim == 1:
      xs = xs.reshape((1, xs.size))
      ys = ys.reshape((1, ys.size))
    print xs.ndim
    return DataSet(xs, ys, self.feature_labels, self.class_labels)
    
  def __getitem__(self, i):
    if isinstance(i, slice):
      return DataSet(self.xs[i, :], self.ys[i,:], self.feature_labels, 
        self.class_labels)
    else:
      if i < 0 or i > self.ninstances: raise ValueError
      return DataSet(
        xs=self.xs[i, :].reshape((1, self.nfeatures)), 
        ys=self.ys[i,:].reshape((1, self.nclasses)), 
        feature_labels=self.feature_labels,
        class_labels=self.class_labels)

  def __len__(self):
    return self.ninstances
 
  def __iter__(self):
    for i in range(self.ninstances):
      yield (self.xs[i], self.ys[i])

  def __str__(self):
    return 'DataSet (%d instances, %d features, classes: %s)' % \
      (self.ninstances, self.nfeatures, repr(self.class_labels))

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
    if a.feature_labels <> b.feature_labels:
      raise ValueError, 'The feature labels do not match'
    if a.class_labels <> b.class_labels:
      raise ValueError, 'The class labels do not match'

    return DataSet(np.vstack([a.xs, b.xs]), np.vstack([a.ys, b.ys]),
      feature_labels=a.feature_labels, class_labels=a.class_labels)

  def __eq__(a, b):
    if isinstance(b, DataSet):
      return (a.xs == b.xs).all() and (a.ys == b.ys).all() and \
        a.feature_labels == b.feature_labels and \
        a.class_labels == b.class_labels
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
  def nfeatures(self):
    if self.xs.ndim == 0:
      return 0
    return self.xs.shape[1]

