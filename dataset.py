import itertools
import numpy as np

class VariableNumberOfFeaturesException(Exception): pass

class DataSet:
  def __init__(self, xs = [], ys = [], signature = ''):
    self.__xs = []
    self.__ys = []
    
    for (x, y) in itertools.izip(xs, ys):
      self.add_instance(x, y)
    
  def add_instance(self, features, label):
    '''Add an instance to this dataset'''
    self.__xs.append(np.array(features))
    self.__ys.append(np.array(label))
    
  def __getitem__(self, i):
    return (self.__xs[i], self.__ys[i])

  def __str__(self):
    state_str = 'DataSet (%d instances, %d features, %dD labels)' % \
      (len(self.__xs), len(self.__xs[0]), len(self.__ys[0]))
    return ' -> '.join(self.signature + [state_str])
  
  def get_xs(self):
    lens = [len(x) for x in self.__xs]
    minl, maxl = min(lens), max(lens)
    if minl <> maxl:
      raise VariableNumberOfFeaturesException
    return np.array(self.__xs)
    
  def get_ys(self):
    return np.array(self.__ys)
    
  def __iter__(self):
    for i in range(self.ninstances):
      yield self[i]
      
  @property
  def labels(self):
    ys = self.get_ys()
    assert(len(ys.shape) == 1)
    return np.unique(ys)
        
  @property
  def ninstances(self):
    return len(self.__xs)

  @property
  def nfeatures(self):
    lens = [len(x) for x in self.__xs]
    minl, maxl = min(lens), max(lens)
    if minl <> maxl:
      raise VariableNumberOfFeaturesException
    return minl
      
