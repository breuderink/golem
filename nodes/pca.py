import logging
import numpy as np
from dataset import *

log = logging.getLogger('PCA')
class PCA:
  def __init__(self):
    self.eigen_cols = None

  def train(self, d):
    self.mean = np.mean(d.xs, axis=0)
    cov = np.cov(d.xs, rowvar=False) 
    U, s, V = np.linalg.svd(cov)
    self.eigen_cols = U
    self.eigen_vals = s
    
    var_explained = np.cumsum(self.eigen_vals) / np.sum(self.eigen_vals)
    log.debug(self.eigen_vals)
    log.debug(var_explained)
  
  def test(self, d):
    xs = d.xs - np.mean(d.xs, axis=0)
    xs = np.dot(xs, self.eigen_cols)
    feature_labels = ['PC%d' % i for i in range(xs.shape[1])]
    return DataSet(xs, d.ys, d.ids, feature_labels, class_labels=d.class_labels)

  def __str__(self):
    W = self.eigen_cols
    if W == None:
      return 'PCA (untrained)'
    return 'PCA (%dD -> %dD)' % (W.shape[1], W.shape[0])
