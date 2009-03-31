# coding: utf8
import logging

import numpy as np
import cvxopt.base as cvx
import cvxopt.solvers

from kernel import *
from ..dataset import DataSet
from ..helpers import hard_max

log = logging.getLogger('SVM')
QP_ACCURACY = 1e-8

class SVM:
  def __init__(self, C=2, kernel=None, hard_max=False, **params):
    self.C = C
    self.kernel = kernel
    self.params = params
    self.hard_max = hard_max

  def train(self, d):
    assert(d.nclasses == 2)
    xs = d.xs
    ys = d.ys

    # See "Learning with Kernels", Schölkopf and Smola, p205
    log.info('Start SVM')
    log.debug('Calculate kernel matrix')
    # linear kernel, so each entry contains the dot-product:
    kernel_matrix = build_kernel_matrix(xs, xs, kernel=self.kernel, 
      **self.params)
    m = xs.shape[0]
    
    log.debug('Creating QP-target')
    # (4) min W(a) = -sum(a_i) + (1/2) * a' * Q * a
    # -sum(a_i) = q'a
    labels = (ys[:,0] - ys[:,1]).reshape(m, 1).astype('d')
    label_matrix = np.dot(labels, labels.T)
    Q = cvx.matrix(kernel_matrix * label_matrix)
    q = cvx.matrix([-1. for i in range(m)])

    log.debug('Creating QP-constraints')
    # (2) 0 < all alphas < C/m, using Ga <= h
    # (3) sum(a_i * y_i) = 0, using Aa = b

    # (2) is solved in two parts, first 0 < alphas, then alphas < C / m
    G1 = cvx.spmatrix(-1, range(m), range(m))
    G2 = cvx.spmatrix(1, range(m), range(m))
    G = cvx.sparse([G1, G2])
    h = cvx.matrix([0. for i in range(m)] + [float(self.C)/m for i in range(m)])
    A = cvx.matrix(labels.T) # The deep copy is needed to prevent a TypeError
    b = cvx.matrix(0.)

    log.debug('Solving QP')
    cvxopt.solvers.options['show_progress'] = False
    cvxopt.solvers.options['abstol'] = QP_ACCURACY
    cvxopt.solvers.options['feastol'] = QP_ACCURACY
    sol = cvxopt.solvers.qp(Q, q, G, h, A, b)
    assert(sol['status'] == 'optimal')
    alphas = np.array(sol['x'])

    log.debug('Extracting Support Vectors')
    # a_i gets close to zero, but does not always equal zero
    sv_ids = np.where(alphas >= QP_ACCURACY)[0]
    log.info('Found %d SVs (%.2f%%)'% (len(sv_ids), len(sv_ids) * 100./m))
    log.debug('Found SVs with indices: ' + str(sv_ids))
    
    model = {}
    model['alphas'] = alphas[sv_ids]
    model['SVs'] = np.array(xs)[sv_ids, :]
    model['labels'] = np.array(labels)[sv_ids, :]
    
    # Calculate b in f(x) = <w, x> + b
    sv_kernel = np.array(kernel_matrix)[sv_ids,:][:, sv_ids]
    model['b'] = np.mean(model['labels'] - 
      np.dot(sv_kernel, (model['labels'] * model['alphas'])).T)
    self.model = model
    self.cl_lab = d.cl_lab

  def test(self, d):
    xs = d.xs
    model = self.model
    SVs, alphas = model['SVs'], model['alphas']
    labels, b = model['labels'], model['b']

    kernel_matrix = build_kernel_matrix(xs, SVs, self.kernel, **self.params)
    # Eq. 7.25: f(x) = sign(sum_i(y_i a_i k(x, x_i) + b)
    labels = np.dot(kernel_matrix, (alphas * labels)) + b

    # Transform into two-colum positive hyperplane distance format
    labels = labels.reshape(-1, 1)
    xs = np.hstack([labels, -labels])
    if self.hard_max:
      xs = hard_max(xs)
    return DataSet(xs, feat_lab=self.cl_lab, default=d)

  def __str__(self):
    return 'SVM (C=%g, kernel=%s, params=%s)' % (self.C, self.kernel, 
      str(self.params))
