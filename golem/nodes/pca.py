import logging
import numpy as np
from ..dataset import DataSet

log = logging.getLogger('PCA')

class PCA:
  def __init__(self, retain=None, ndims=None):
    self.eigen_cols = None
    self.retain = retain
    self.ndims = ndims

  def train(self, d):
    self.mean = np.mean(d.xs, axis=0)
    cov = np.cov(d.xs, rowvar=False) 
    U, s, V = np.linalg.svd(cov)
    self.eigen_cols = U
    self.eigen_vals = s
    
    if self.ndims <> None:
      self.eigen_cols = self.eigen_cols[:, :self.ndims]
    elif self.retain <> None:
      var_explained = np.cumsum(self.eigen_vals) / np.sum(self.eigen_vals)
      last_component = np.where(var_explained >= self.retain)[0][0]
      self.eigen_cols = self.eigen_cols[:, :last_component + 1]
  
  def test(self, d):
    xs = d.xs - np.mean(d.xs, axis=0)
    xs = np.dot(xs, self.eigen_cols)
    feature_labels = ['PC%d' % i for i in range(xs.shape[1])]
    return DataSet(xs, feat_lab=feature_labels, default=d)

  def __str__(self):
    W = self.eigen_cols
    if W == None:
      return 'PCA (untrained)'
    return 'PCA (%dD -> %dD)' % (W.shape[0], W.shape[1])
