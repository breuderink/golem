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

class TestKullbackLeibler(unittest.TestCase):
  def setUp(self):
    A = np.random.randn(4, 4)
    self.Sig1 = np.dot(A, A.T)
    self.inv_Sig1 = np.linalg.inv(self.Sig1)
    self.mu1 = np.random.randn(4)

    B = np.random.randn(4, 4)
    self.Sig2 = np.dot(B, B.T)
    self.inv_Sig2 = np.linalg.inv(self.Sig2)
    self.mu2 = np.random.randn(4)

  def test_equal_dist(self):
    Sig_p, inv_Sig_p, mu = self.Sig1, self.inv_Sig1, self.mu1
    self.assertAlmostEqual(norm_kl_divergence(inv_Sig_p, mu, Sig_p, mu), 0)

  def test_mean_divergence(self):
    Sig_q, inv_Sig_p, mu = self.Sig1, self.inv_Sig1, self.mu1

    for i in range(4):
      # generate random direction
      rd = np.random.randn(Sig_q.shape[0])

      # shift one mean in this direction
      kld = np.asarray([norm_kl_divergence(inv_Sig_p, mu, Sig_q, mu + rd * d)
        for d in np.linspace(0, 10, 50)])
      
      # check that the KLD is monotonically increasing
      self.assert_(np.all(np.diff(kld) > 0))

  def test_cov_divergence(self):
    Sig_q, inv_Sig_p, mu = self.Sig1, self.inv_Sig1, self.mu1
    Sig_p = self.Sig2
    kl = []
    for alpha in np.linspace(0, 1, 10):
      # create diverging covariance matrix
      S = alpha * Sig_p + (1. - alpha) * Sig_q
      kl.append(norm_kl_divergence(inv_Sig_p, mu, S, mu))

    self.assert_(np.all(np.diff(kl) > 0))

  def test_numerical(self):
    mu_p, mu_q, sig_p, sig_q = -1, 0, 1, .5
    kld_an = norm_kl_divergence(sig_p, mu_p, 1./sig_q, mu_q)

    def norm_pdf(x, mu, sig):
      return 1./np.sqrt(np.pi * 2. * sig **2)* np.exp(-(x-mu)**2./(2.*sig**2))

    xs = np.linspace(-10, 10, 5000)
    px = norm_pdf(xs, mu_p, sig_p**.5)
    qx = norm_pdf(xs, mu_q, sig_q**.5)
    div = px * np.log(px/qx)
    kld_num = np.trapz(div, xs)

    np.testing.assert_almost_equal(kld_num, kld_an)
