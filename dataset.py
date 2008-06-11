import itertools
import numpy as np

# @@TODO:
# - add support for ids (for time etc)
# - what about 2d and 3d instances?
# - labels vs regression
# - is it logical to convert everything to np.array? -> convert everything to
# np.array when needed.
# - .get_xs -> rename to get_xs_array?
# - add label names, to be used for example in plots


class VariableNumberOfFeaturesException(Exception): pass

class DataSet:
  def __init__(self, xs = [], ys = []):
    '''Create a new dataset.'''
    if len(xs) <> len(ys):
      raise ValueError
    
    self.__xs = []
    self.__ys = []
    
    for (x, y) in itertools.izip(xs, ys):
      self.add_instance(x, y)
    
  def add_instance(self, features, label):
    '''Add an instance to this dataset'''
    self.__xs.append(features);
    self.__ys.append(label);
    

  def get_xs(self):
    '''Return the features as a numpy-array'''
    if not self.const_feature_len():
      raise VariableNumberOfFeaturesException
    return np.array(self.__xs)
    
  def get_ys(self):
    '''Return the labels as a numpy-array'''
    return np.array(self.__ys)
    
  def const_feature_len(self):
    lens = [len(x) for x in self.__xs]
    return min(lens) == max(lens)

  def __getitem__(self, i):
    return (self.__xs[i], self.__ys[i])
 
  def __iter__(self):
    for i in range(self.ninstances):
      yield self[i]

  def __str__(self):
    state_str = 'DataSet (%d instances, %d features, %d unique labels)' % \
      (self.ninstances, self.nfeatures, len(self.labels))
    return state_str
    
  @property
  def labels(self):
    ys = self.get_ys()
    non_singleton_dims = [d for d in ys.shape if d > 1]
    assert(len(non_singleton_dims) == 1)
    return np.unique(ys)
        
  @property
  def ninstances(self):
    return len(self.__xs)

  @property
  def nfeatures(self):
    if not self.const_feature_len():
      raise VariableNumberOfFeaturesException
    return len(self.__xs[0])
