import numpy as np

def lw_cov(X, center=True):
  '''
  Calculate the well-conditioned (it is invertible) and accurate (it is
  asymptotically more accurate than the sample covariance matrix) Ledoit-Wolf
  covariance estimator [1].

  For X, a [n x p] matrix, where n is the number of observations, and p is the
  number of variables, a robust covariance estimate is returned.  It
  analytically estimates the optimial weight \lambda^{\star} for a weigheted
  combination of the sample mean S and and a prior of equal variance and zero
  covariance: 
  
    S^{\star} = \lambda^{\star} \S  + (1-\lambda^{\star}) * \sigma * I

  See (14) of [1]. \lambda^star{\star} is based on the variation in the
  covariance and the difference between the prior and and the sample covariance
  S.

  [1] Olivier Ledoit and Michael Wolf. A well-conditioned estimator for
      large-dimensional covariance matrices. Journal of Multivariate Analysis,
      88(2):365--411, February 2004.
  '''
  X = np.asarray(X)
  n, t = X.shape
  if center:
    X = X - np.mean(X, axis=0)

  S = np.dot(X.T, X) / n
  prior = np.mean(np.diag(S)) * np.eye(t) # m * I

  b2, d2, lamb = lw_cov_base(X, S, prior)

  return lamb * prior + (1.-lamb) * S

def lw_cov_base(X0, S, prior):
  '''
  Calculate \lambda^{\star}, d^2 and b^2 for Ledoit-Wolf covariance estimator.
  Separate method for unit-testing.
  '''
  n, t = X0.shape

  # Calculate  \delta^2 using Lemma 3.3
  d2 = np.linalg.norm(S - prior, 'fro') ** 2

  # Calculate \bar{\beta}^2 as in Lemma 3.4, but using
  # var(x) = E(x^2) - [E(x)]^2:
  XoX = X0**2
  varS = np.dot(XoX.T, XoX) / n - S**2
  b2 = np.sum(varS) / n

  # Calculate shrinkage intensity
  lamb = np.clip(b2/d2, 0, 1)
  return b2, d2, lamb

def norm_kl_divergence(Sig_p, mu_p, inv_Sig_q, mu_q):
  '''
  Calculate Kullback-Leibler divergence between two multivariate *normal*
  distributions analytically, specified by covariance Sig_p and mean mu_p and 
  inverse covariance inv_Sig_q and mean mu_q:

    KL(P || Q) = \int_x p(x) \log(p(x) / q(x)) dx

  Please note that for the Q distribution the *INVERSE COVARIANCE* has to be
  specified.
  '''
  Sig_p, mu_p, inv_Sig_q, mu_q = np.atleast_2d(Sig_p, mu_p, inv_Sig_q, mu_q)
  assert Sig_p.shape == inv_Sig_q.shape
  assert mu_p.size == mu_q.size == Sig_p.shape[0]

  A = np.dot(Sig_p, inv_Sig_q)
  B = np.trace(np.eye(mu_q.size) - A)
  C = reduce(np.dot, [(mu_p - mu_q), inv_Sig_q, (mu_p - mu_q).T])
  return -.5 * (np.log(np.linalg.det(A)) + B - C).squeeze()

def kl(P, Q):
  '''
  Return the Kullack-Leibler divergence between two distributions P and Q
  using norm_kl_divergence(). Please note that the inversion of the covariance
  matrix of Q (\Sigma_Q) might cause numerically unstable results although
  lw_cov() is used to estimate \Sigma_Q.

  The Kullback-Leibler divergence is an asymmetric distance measure between two
  probability measures P and Q, measuring the extra information needed to repre-
  sent samples from P when using a code based on Q.
  '''
  S_p, m_p = lw_cov(P), np.mean(P, 0)
  S_q, m_q = lw_cov(Q), np.mean(Q, 0)
  return norm_kl_divergence(S_p, m_p, np.linalg.pinv(S_q), m_q)
