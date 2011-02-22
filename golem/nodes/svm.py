# coding: utf8
import logging
import numpy as np
import cvxopt.base as cvx
import cvxopt.solvers

from basenode import BaseNode
from ..kernel import build_kernel_matrix
from ..dataset import DataSet

ALPHA_RTOL = 1e-5

def cvxopt_svm(K, labels, C):
  # See "Learning with Kernels", Schölkopf and Smola, p.205
  log = logging.getLogger('golem.cvxopt_svm')
  m = K.shape[0]
  labels = np.atleast_2d(labels)

  assert np.all(np.unique(labels) == [-1, 1])
  assert K.shape[0] == K.shape[1]
  
  log.debug('Creating QP-target')
  # (4) min W(a) = -sum(a_i) + (1/2) * a' * Q * a
  # -sum(a_i) = q'a
  label_matrix = np.dot(labels.T, labels)
  Q = cvx.matrix(K * label_matrix)
  q = cvx.matrix([-1. for i in range(m)])

  log.debug('Creating QP-constraints')
  # (2) 0 < all alphas < C/m, using Ga <= h
  # (3) sum(a_i * y_i) = 0, using Aa = b
  # (2) is solved in two parts, first 0 < alphas, then alphas < C / m
  G1 = cvx.spmatrix(-1, range(m), range(m))
  G2 = cvx.spmatrix(1, range(m), range(m))
  G = cvx.sparse([G1, G2])
  h = cvx.matrix([0. for i in range(m)] + [float(C)/m for i in range(m)])
  A = cvx.matrix(labels)
  b = cvx.matrix(0.)

  log.debug('Solving QP')
  cvxopt.solvers.options['show_progress'] = False
  sol = cvxopt.solvers.qp(Q, q, G, h, A, b)
  if sol['status'] != 'optimal':
    log.warning('QP solution status: ' + sol['status'])
  log.debug('solver.status = ' + sol['status'])
  alphas = np.array(sol['x'])

  # a_i gets close to zero, but does not always equal zero:
  alphas[alphas < np.max(alphas) * ALPHA_RTOL] = 0
  return alphas.flatten()

class SVM(BaseNode):
  def __init__(self, C=2, kernel=None, **params):
    BaseNode.__init__(self)
    self.C = C
    self.kernel = kernel
    self.params = params

  def train_(self, d):
    assert d.nclasses == 2
    self.log.debug('Calculate kernel matrix')
    K = build_kernel_matrix(d.X, d.X, kernel=self.kernel, 
      **self.params)
    labels = np.atleast_2d((d.ys[:, 0] - d.ys[:, 1])).astype(float)
    alphas = np.atleast_2d(cvxopt_svm(K, labels, self.C))
    
    # Calculate b in f(x) = wx + b
    b = np.mean(labels.T - np.dot(labels * alphas, K))

    # Extract support vectors
    svs = d[alphas.flat != 0]
    self.log.info('Found %d SVs (%.2f%%)' % (svs.ninstances, 
      svs.ninstances * 100./d.ninstances))

    # Store model
    self.alphas = alphas
    self.svs = svs
    self.b = b
    self.w = (labels * alphas)[:, alphas > 0]

  def apply_(self, d):
    # Eq. 7.25: f(x) = sign(sum_i(y_i a_i k(x, x_i) + b), we do not sign!
    K = build_kernel_matrix(self.svs.X, d.X, self.kernel, **self.params)
    preds = np.atleast_2d(np.dot(self.w, K) + self.b)

    # Transform into two-column hyperplane distance format
    return DataSet(np.hstack([preds.T, -preds.T]), default=d)

  def __str__(self):
    return 'SVM (C=%g, kernel=%s, params=%s)' % (self.C, self.kernel, 
      str(self.params))
