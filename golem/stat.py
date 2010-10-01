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
  Used for unit-testing.
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

