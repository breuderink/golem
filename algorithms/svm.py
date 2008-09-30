# coding: utf8
import logging
import pickle
import math

import numpy as np
import pylab
import cvxopt.base as cvx
import cvxopt.solvers

from kernel import *
from helpers.plots import *

log = logging.getLogger('SVM')
QP_ACCURACY = 1e-8

class SupportVectorMachine:
  def __init__(self, C=2, kernel=None, sign_output=True, **params):
    self.C = C
    self.kernel = kernel
    self.params = params
    self.sign_output = sign_output

  def train(self, xs, ys):
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
    label_matrix = np.dot(ys, ys.T).astype('d') # Could be made sparse?
    Q = cvx.matrix(kernel_matrix * label_matrix) # Only lower triangular part is needed
    q = cvx.matrix([-1. for i in range(m)])

    log.debug('Creating QP-constraints')
    # (2) 0 < all alphas < C/m, using Ga <= h
    # (3) sum(a_i * y_i) = 0, using Aa = b

    # (2) is solved in two parts, first 0 < alphas, then alphas < C / m
    G1 = cvx.spmatrix(-1, range(m), range(m))
    G2 = cvx.spmatrix(1, range(m), range(m))
    G = cvx.sparse([G1, G2])
    h = cvx.matrix([0. for i in range(m)] + [float(self.C)/m for i in range(m)])
    A = cvx.matrix(ys.T.copy()) # The deep copy is needed to prevent a TypeError
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
    model['ys'] = np.array(ys)[sv_ids, :]
    
    # Calculate b in f(x) = <w, x> + b
    sv_kernel = np.array(kernel_matrix)[sv_ids,:][:, sv_ids]
    model['b'] = np.mean(
      model['ys'] - np.dot(sv_kernel, (model['ys'] * model['alphas'])).T)
    self.model = model

  def test(self, xs):
    model = self.model
    SVs, alphas, ys, b = model['SVs'], model['alphas'], model['ys'], model['b']
    kernel_matrix = build_kernel_matrix(xs, SVs, 
      self.kernel, **self.params)
    # Eq. 7.25: f(x) = sign(sum_i(y_i a_i k(x, x_i) + b)
    ys = np.dot(kernel_matrix, (alphas * ys)) + b

    # Transform into two-colum positive hyperplane distance format
    ys = ys.reshape(ys.size, 1)
    ys = np.hstack([ys, -ys])
    if self.sign_output:
      ys = np.sign(ys)
    return ys

