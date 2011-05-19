#coding: utf8
import logging, warnings
import numpy as np
import cvxopt.base as cvx
import cvxopt.solvers

from basenode import BaseNode
from ..kernel import build_kernel_matrix, kernel_cv_fold
from ..helpers import hard_max
from ..dataset import DataSet

ALPHA_RTOL = 1e-5

def cvxopt_svm(K, labels, c):
  # See "Learning with Kernels", Schölkopf and Smola, p.205
  log = logging.getLogger('golem.nodes.svm.cvxopt_svm')
  c = float(c)
  m = K.shape[0]
  labels = np.atleast_2d(labels)

  assert np.all(np.unique(labels) == [-1, 1])
  assert K.shape[0] == K.shape[1]
  
  log.debug('Creating QP-target')
  # (4) min W(a) = (1/2) * a^T * P * a - \vec{1}^T a
  label_matrix = np.dot(labels.T, labels)
  P = cvx.matrix(K * label_matrix)
  q = cvx.matrix([-1. for i in range(m)])

  log.debug('Creating QP-constraints')
  # (2) 0 < all alphas < c/m, using Ga <= h
  # (3) sum(a_i * y_i) = 0, using Aa = b
  # (2) is solved in two parts, first 0 < alphas, then alphas < c / m
  G1 = cvx.spmatrix(-1, range(m), range(m))
  G2 = cvx.spmatrix(1, range(m), range(m))
  G = cvx.sparse([G1, G2])
  h = cvx.matrix([0. for i in range(m)] + [c/m for i in range(m)])
  A = cvx.matrix(labels)
  r = cvx.matrix(0.)

  log.debug('Solving QP')
  cvxopt.solvers.options['show_progress'] = False
  sol = cvxopt.solvers.qp(P, q, G, h, A, r)
  if sol['status'] != 'optimal':
    log.warning('QP solution status: ' + sol['status'])
  log.debug('solver.status = ' + sol['status'])
  alphas = np.asarray(sol['x']) # column vector!
  
  bias = np.mean(labels.flatten() - np.dot(labels.flatten() * alphas.T, K))
  # alpha_i gets close to zero, but does not always equal zero:
  alphas[alphas < np.max(alphas) * ALPHA_RTOL] = 0
  return alphas.flatten(), bias

class SVM(BaseNode):
  def __init__(self, c=np.logspace(-3, 5, 10), kernel=None, **params):
    BaseNode.__init__(self)
    self.c = np.atleast_1d(c)
    self.c_star = np.nan
    self.kernel = kernel
    self.kernel_params = params
    self.nfolds = 5

    if 'C' in params.keys():
      warnings.warn(
        "The SVM's C-parameter has been replaced with a lowercase c.",
        DeprecationWarning)
      self.c = np.atleast_1d(params['C'])

  def train_(self, d):
    assert d.nclasses == 2
    self.log.debug('Calculating kernel matrix')
    K = build_kernel_matrix(d.X, d.X, kernel=self.kernel, 
      **self.kernel_params)
    labels = np.where(d.Y[1] > d.Y[0], 1, -1).astype(float)

    # find c-parameter
    if self.c.size == 1:
      self.c_star = self.c[0]
    else:
      accs = []
      folds = np.arange(d.ninstances) % self.nfolds
      for c in self.c:
        self.log.debug('Evaluating c=%.3g' % c)
        preds = svm_crossval(K, labels, c, folds)
        accs.append(np.mean(labels == np.sign(preds)))
        self.log.debug('CV accuracy: %.3f' % accs[-1])
      self.c_star = self.c[np.argmax(accs)]
      self.log.info('Selected c=%.3g' % self.c_star)

    # train final SVM
    alphas, b = cvxopt_svm(K, labels, self.c_star)

    # extract support vectors
    svs = d[alphas != 0]
    self.log.info('Found %d SVs (%.2f%%)' % (svs.ninstances, 
      svs.ninstances * 100./d.ninstances))

    # store (sparse) model
    self.alphas = alphas
    self.svs = svs
    self.b = b
    self.sv_weights = (labels * alphas)[:, alphas > 0]

  def apply_(self, d):
    # eq. 7.25: f(x) = sign(sum_i(y_i a_i k(x, x_i) + b), but we keep distance.
    K = build_kernel_matrix(self.svs.X, d.X, self.kernel, **self.kernel_params)
    preds = np.atleast_2d(np.dot(self.sv_weights, K) + self.b)

    # transform into two-column hyperplane distance format
    return DataSet(X=np.vstack([-preds, preds]), default=d)

  def __str__(self):
    return 'SVM (c=%g, kernel=%s, params=%s)' % (self.c_star, self.kernel, 
      str(self.kernel_params))


def svm_crossval(K, labels, c, folds):
  ''' Low-level SVM cross-validation procedure '''
  preds = np.zeros(K.shape[0]) * np.nan
  for fi in np.unique(folds):
    # get kernels and labels for fold
    K_tr, K_te = kernel_cv_fold(K, folds, fi)
    tr_lab = labels[folds!=fi]

    # train SVM
    alphas, b = cvxopt_svm(K_tr, tr_lab, c)
    preds[folds==fi] = np.dot(alphas * tr_lab, K_te) + b
  return preds
