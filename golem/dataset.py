import itertools
import numpy as np
import helpers

class DataSet:
  def __init__(self, xs, ys, ids, class_labels=None, feature_labels=None, 
    feature_shape=None):
    '''Create a new dataset.'''
    if xs == None and ys == None:
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

    self.ids = np.asarray(ids).reshape(-1, 1) if ids <> None\
      else np.arange(self.ninstances).reshape(-1, 1)
    assert(np.unique(self.ids).size == self.ids.size)

    self.feature_shape = feature_shape

  def get_class(self, i):
    return self[self.ys[:, i] == 1]

  def sort(self):
    '''Sort by id'''
    return self[np.argsort(self.ids.flatten())]
    
  def __getitem__(self, i):
    if isinstance(i, slice) or isinstance(i, list) or isinstance(i, np.ndarray):
      return DataSet(
        xs=self.xs[i, :], ys=self.ys[i,:], ids=self.ids[i, :], 
        feature_labels=self.feature_labels, class_labels=self.class_labels)
    elif isinstance(i, int):
      return DataSet(
        xs=self.xs[i, :].reshape(1, -1),
        ys=self.ys[i,:].reshape(1, -1),
        ids=self.ids[i,:].reshape(1, -1),
        feature_labels=self.feature_labels,
        class_labels=self.class_labels)
    else:
      raise ValueError, 'Unkown indexing type.'

  def nd_xs(self):
    '''Return N-dimensional view of xs'''
    if self.feature_shape <> None:
      return self.xs.reshape([self.ninstances] + self.feature_shape)
    raise Exception, 'Feature shape is unknown'

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
    if a.feature_labels <> b.feature_labels:
      raise ValueError, 'The feature labels do not match'
    if a.class_labels <> b.class_labels:
      raise ValueError, 'The class labels do not match'

    return DataSet(np.vstack([a.xs, b.xs]), np.vstack([a.ys, b.ys]),
      feature_labels=a.feature_labels, class_labels=a.class_labels, 
      ids=np.vstack([a.ids, b.ids]))

  def __eq__(a, b):
    if isinstance(b, DataSet):
      return (a.xs == b.xs).all() and (a.ys == b.ys).all() and \
        a.feature_labels == b.feature_labels and \
        a.class_labels == b.class_labels and\
        (a.ids == b.ids).all()
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
