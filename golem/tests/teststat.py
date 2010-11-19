import unittest
import numpy as np
from ..stat import *
from ..helpers import to_one_of_n

class TestLedoitWolfCov(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    p, n = 40, 50
    self.A = A = np.random.randn(p, p)
    self.Sigma = np.dot(A, A.T)
    X = np.random.randn(p, n)
    X -= np.atleast_2d(np.mean(X, 1)).T
    X = np.dot(A, X)
    self.X0 = X0 = X - np.atleast_2d(np.mean(X, 1)).T
    self.S = np.dot(X0, X0.T) / n

  def test_var_of_cov(self):
    X0, S = self.X0, self.S
    p, n = X0.shape

    V = np.mean(
      [(np.dot(np.atleast_2d(o).T, np.atleast_2d(o)) - S)**2 for o in X0.T], 
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

  def test_convenience_fun(self):
    P = np.dot(np.random.randn(4, 4), np.random.rand(4, 10))
    Q = np.dot(np.random.randn(4, 4), np.random.rand(4, 100))

    self.assertAlmostEqual(
      kl(P, Q),
      norm_kl_divergence(lw_cov(P), np.mean(P, 1), 
        np.linalg.pinv(lw_cov(Q)), np.mean(Q, 1)))

class TestROC(unittest.TestCase):
  def test_roc(self):
    '''Test bounds and ordering of ROC'''
    TPs, FPs = roc(np.random.rand(100), np.random.rand(100).round())
    # test mononely increasing TPs and FPs
    np.testing.assert_equal(np.sort(TPs), TPs)
    np.testing.assert_equal(np.sort(FPs), FPs)
    
    self.assertEqual(TPs.min(), 0)
    self.assertEqual(TPs.max(), 1)
    self.assertEqual(FPs.min(), 0)
    self.assertEqual(FPs.max(), 1)

  def test_reverse(self):
    '''Test that the ROC is invariant for reversions'''
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)
    t1, f1 = roc(scores[::-1], labels[::-1]) # reversed ROC
    np.testing.assert_equal(t0, t1)
    np.testing.assert_equal(f0, f1)
  
  def test_known(self):
    '''Test ROC for known input'''
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)

    self.assert_((t0 == [0, .25, 1, 1]).all())
    self.assert_((f0 == [0, 0, .75, 1]).all())

class TestAUC(unittest.TestCase):
  def test_AUC_extrema(self):
    '''Test AUC for extrema'''
    self.assertEqual(auc([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]), 1)
    self.assertEqual(auc([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]), 0)
    self.assertEqual(auc([1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0]), .5)

  def test_AUC_symbounds(self):
    '''Test AUC for symmetry and bounds'''
    N = 100
    for rho in [.1, .3, .5]:
      for i in range(20):
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertAlmostEqual(auc(xs, ys), 1-auc(xs, np.abs(ys-1)))
        self.assert_(0 <= auc(xs, ys) <= 1)

  def test_AUC_confidence(self):
    '''Test AUC confidence interval for trends'''
    # we do not know much, but we can test for trends
    self.assert_(auc_confidence(1) > auc_confidence(100))
    self.assert_(auc_confidence(100, rho=.1) > auc_confidence(100))
    self.assert_(auc_confidence(100, delta=1e-8) > auc_confidence(100))

    # and symmetry
    for rho in [.01, .1, .5]:
      self.assertAlmostEqual(auc_confidence(100, rho=rho),
        auc_confidence(100, rho=1-rho))

  def test_monte_carlo(self):
    '''Monte Carlo test for AUC confidence intervals'''
    SAMPLES = 100
    for N in [10, 100, 1000]:
      for rho in [0.1, .5, .9]:
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertEqual(ys.mean(), rho)
        aucs = []

        # create random AUCs
        for i in range(SAMPLES):
          np.random.shuffle(ys)
          aucs.append(auc(xs, ys))

        # test conservativeness
        for delta in [.05, .001, .0001]:
          epsilon = auc_confidence(N, rho, delta)
          dev = np.abs(np.array(aucs) - 0.5)
          e_p = np.mean(dev > epsilon)
          self.assert_(e_p <= delta, 
            'empirical p (=%f) > delta (=%f)' % (e_p, delta))

class TestMutualInformation(unittest.TestCase):
  def test_max_bits(self):
    for i in range(4):
      conf = np.eye(2 ** i)
      self.assertAlmostEqual(mut_inf(conf), i)

  def test_uniform(self):
    for i in range(4):
      conf = np.ones((i, i + 1))
      self.assertAlmostEqual(mut_inf(conf), 0)

  def test_zero(self):
    self.assert_(np.isnan(mut_inf(np.zeros((5, 3)))))

  def test_no_modification(self):
    conf = np.ones((4, 3))
    mut_inf(conf)
    np.testing.assert_equal(conf, np.ones((4, 3)))

  def test_symmetrical(self):
    for i in range(4):
      conf = np.random.rand(3, 8)
      self.assertAlmostEqual(mut_inf(conf), mut_inf(conf.T))

  def test_malformed(self):
    self.assertRaises(AssertionError, mut_inf, -np.ones((3, 3)))
