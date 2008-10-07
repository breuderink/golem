import itertools
import numpy as np

# - add support for ids (for time etc)

class DataSet:
  def __init__(self, xs=None, ys=None):
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

    self.feature_labels = ['feat%d' % i for i in range(self.nfeatures)]
    self.class_labels = ['class_%d' % i for i in range(self.nclasses)]

    
  def __getitem__(self, i):
    if isinstance(i, slice):
      return DataSet(self.xs[i, :], self.ys[i,:])
    else:
      if i < 0 or i > self.ninstances: raise ValueError
      return DataSet(self.xs[i, :].reshape((1, self.nfeatures)), 
        self.ys[i,:].reshape((1, self.nclasses)))

  def __len__(self):
    return self.ninstances
 
  def __iter__(self):
    for i in range(self.ninstances):
      yield (self.xs[i], self.ys[i])

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

  def __eq__(a, b):
    if isinstance(b, DataSet):
      return (a.xs == b.xs).all() and (a.ys == b.ys).all()
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

