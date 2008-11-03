import logging
import numpy as np
from golem import DataSet

log = logging.getLogger('CSP')

# TODO:
# 1 Use Fisher criterion for feature selection
class CSP:
  def __init__(self, m=2, axis=-1):
    self.W = None
    self.m = m
    assert(axis in [-1, 0])
    self.axis = axis

  def axis_xs(self, d):
    '''Create 2D array to work with depending on axis'''
    if self.axis <> -1:
      xs = np.concatenate(d.nd_xs, axis=self.axis)
    else:
      xs = d.xs
    assert(xs.ndim == 2)
    return xs

  def train(self, d):
    assert(d.nclasses == 2)
    
    # Store mean
    xs = self.axis_xs(d)
    self.mean = np.mean(xs, axis=0)

    # Calc whitening matrix P
    cov = np.cov(xs, rowvar=False) 
    log.debug('Cov shape: %s' % str(cov.shape))
    U, s, V = np.linalg.svd(cov)
    P = np.dot(U, np.linalg.pinv(np.diag(s)) ** (.5))
    self.P = P[:, :np.rank(P)]
    

    # Calc class-diagonalization matrix B
    d0 = d.get_class(0)
    xs0 = np.dot(self.axis_xs(d0), self.P)
    self.B, s0, V0 = np.linalg.svd(np.cov(xs0, rowvar=False))

    # Construct final transormation matrix W
    self.W = np.dot(self.P, self.B)
    if self.W.shape[1] > self.m * 2:
      comps = range(self.m) + range(-1, -self.m -1, -1)
      log.debug('Selecting components %s' % comps)
      self.W = self.W[:, comps]
    else:
      log.warning('Rank to low to select %d components.' % self.m)

  def test(self, d):
    xs = d.xs - np.mean(d.xs, axis=0)
    xs = np.dot(xs, self.W)
    feature_labels = ['CSP_Comp%d' % i for i in range(xs.shape[1])]
    return DataSet(xs, d.ys, d.ids, feature_labels=feature_labels, 
      class_labels=d.class_labels)

  def __str__(self):
    W = self.eigen_cols
    if W == None:
      return 'CSP (untrained)'
    return 'CSP (%dD -> %dD)' % (W.shape[0], W.shape[1])
