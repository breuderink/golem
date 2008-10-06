import itertools
import numpy as np

# Make this class a lot simpeler. Complexity can always added later.
# For now make a basic DataSet class, with no support for variable length
# features. To support segments create a derived class. Also make the interface
# simpeler; only NP arrays. This should shorten the length of unittests.  

# - add support for ids (for time etc)


class VariableNumberOfFeaturesException(Exception): pass

class DataSet:
  def __init__(self, xs=None, ys=None):
    '''Create a new dataset.'''
    if xs == None:
      xs = np.array(None)
      ys = np.array(None)
    elif not (isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray)):
      raise ValueError, 'Only numpy.ndarray is supported for xs and ys.'
    elif xs.ndim <> 2 or ys.ndim <> 2:
      raise ValueError, 'Incorrect number of dimensions for xs or ys'
    elif xs.shape[0] <> ys.shape[0]:
      raise ValueError, 'The #instances does not match #labels'
   
    self.xs = xs
    self.ys = ys

    self.feature_labels = ['feat%d' % i for i in range(self.nfeatures)]
    self.class_labels = ['class_%d' % i for i in range(self.nclasses)]

    
  def __getitem__(self, i):
    return (self.xs[i, :], self.ys[i,:])
 
  def __iter__(self):
    for i in range(self.ninstances):
      yield self[i]

  def __str__(self):
    state_str = 'DataSet (%d instances, %d features, %d classes)' % \
      (self.ninstances, self.nfeatures, self.nclasses)
    return state_str

  def __add__(a, b):
    '''Create a new DataSet by adding the instances of b to a'''
    assert(isinstance(a, DataSet))
    assert(isinstance(b, DataSet))

    # Handle empty datasets
    if a.xs.ndim == 0:
      return b
    if b.xs.ndim == 0:
      return a
    # Check shape
    if (a.nfeatures <> b.nfeatures) or (a.nclasses <> b.nclasses):
      raise ValueError, 'The #features or #classes do not match'
    return DataSet(np.vstack([a.xs, b.xs]), np.vstack([a.ys, b.ys]))
    
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

