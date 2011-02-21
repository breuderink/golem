import logging
import numpy as np
from basenode import BaseNode
from ..dataset import DataSet
from ..stat import lw_cov

class PCA(BaseNode):
  ''' 
  PCA node
  '''
  def __init__(self, retain=.95, ndims=None, cov_f=lw_cov):
    BaseNode.__init__(self)
    self.retain = float(retain)
    self.ndims = ndims
    self.cov_f = cov_f

  def train_(self, d):
    self.mean = np.mean(d.X, axis=1).reshape(-1, 1)
    _, eigvals, W = np.linalg.svd(self.cov_f(d.X))
    
    if self.ndims != None:
      self.W = W[:self.ndims]
    elif self.retain != None:
      var_explained = np.cumsum(eigvals) / np.sum(eigvals)
      self.W = W[:np.sum(var_explained < self.retain)+1]
    self.log.info('Selected %d principal components.' % self.W.shape[0])
  
  def apply_(self, d):
    X = np.dot(self.W, d.X - self.mean)
    feature_labels = ['PC%d' % i for i in range(self.W.shape[0])]
    return DataSet(X=X, feat_lab=feature_labels, default=d)

  def __str__(self):
    if hasattr(self, 'W'):
      return 'PCA (%dD -> %dD)' % (self.W.T.shape)
    return 'PCA'
