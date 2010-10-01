import unittest
import numpy as np
from ..stat import *

class TestLedoitWolfCov(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    N, P = 50, 40
    self.A = A = np.random.randn(P, P)
    self.Sigma = np.dot(A.T, A)
    X = np.random.randn(N,  P)
    X -= np.mean(X, axis=0)
    X = np.dot(X, A)
    self.X0 = X0 = X - np.mean(X, axis=0)
    self.S = np.dot(X0.T, X0) / N

  def test_var_of_cov(self):
    X0, S = self.X0, self.S
    n, p = X0.shape

    V = np.mean(
      [(np.dot(np.atleast_2d(o).T, np.atleast_2d(o)) - S)**2 for o in X0], 
      axis=0)

    b2, d2, lamb = lw_cov_base(X0, S, np.eye(p))
    self.assertAlmostEqual(np.sum(V) / n, b2)

  def test_condition_number(self):
    S_star = lw_cov(self.X0)
    S = np.cov(self.X0, rowvar=False)
    self.assert_(np.linalg.cond(S_star) < np.linalg.cond(S))

  def test_accuracy(self):
    X, S, Sigma = self.X0, self.S, self.Sigma
    Sigma = np.dot(self.A.T, self.A)
    self.assert_(np.linalg.norm(lw_cov(X) - Sigma)
      < np.linalg.norm(S - Sigma))

  def test_inv_accuracy(self):
    X, S, Sigma = self.X0, self.S, self.Sigma
    S_star = lw_cov(X)
    invSigma, invS, invS_star = [np.linalg.inv(Y) for Y in [Sigma, S, S_star]]
    self.assert_(np.linalg.norm(invS_star - invSigma) 
      < np.linalg.norm(invS - invSigma))

if __name__ == '__main__':
  unittest.main()
