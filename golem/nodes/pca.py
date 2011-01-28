import logging
import numpy as np
from basenode import BaseNode
from ..dataset import DataSet

class PCA(BaseNode):
  ''' 
  PCA node
  '''
  def __init__(self, retain=.95, ndims=None):
    BaseNode.__init__(self)
    self.retain = retain
    self.ndims = ndims

  def train_(self, d):
    self.mean = np.mean(d.xs, axis=0)
    cov = np.cov(d.xs, rowvar=False) 
    U, s, V = np.linalg.svd(cov)
    self.eigen_cols = U
    self.eigen_vals = s

    self.log.debug('Eigen values: %s.' % self.eigen_vals)
    
    if self.ndims != None:
      self.eigen_cols = self.eigen_cols[:, :self.ndims]
    elif self.retain != None:
      var_explained = np.cumsum(self.eigen_vals) / np.sum(self.eigen_vals)
      last_component = np.where(var_explained >= self.retain)[0][0]
      self.eigen_cols = self.eigen_cols[:, :last_component + 1]

    self.log.info('Selected %d components.' % self.eigen_cols.shape[1])
  
  def apply_(self, d):
    xs = d.xs - np.mean(d.xs, axis=0)
    xs = np.dot(xs, self.eigen_cols)
    feature_labels = ['PC%d' % i for i in range(xs.shape[1])]
    return DataSet(xs, feat_lab=feature_labels, default=d)

  def __str__(self):
    if hasattr(self, 'eigen_cols'):
      return 'PCA (%dD -> %dD)' % (self.eigen_cols.shape)
    return 'PCA'
