import operator
import numpy as np
from scipy import stats
from .. import DataSet
from basenode import BaseNode
from ..stat import lw_cov

class RDA(BaseNode):
  def __init__(self, alpha=.0, beta=.3, cov_f=lw_cov):
    '''
    Regularized Discriminant Analysis, Alpaydin, p.98, Eq. 5.29:
    S_i^{'} = \alpha \sigma^2I + \beta S + (1 - \alpha - \beta)S_i
    
    alpha = beta = 0 results in a quadratic classfier,
    alpha = 0, beta = 1 results in a linear classifier,
    alpha = 1, beta = 0 results in a nearest mean classifier.
    '''
    BaseNode.__init__(self)
    self.alpha = float(alpha)
    self.beta = float(beta)
    self.cov_f = cov_f

  def train_(self, d):
    # estimate means and covariance matrices
    covs = [self.cov_f(d.get_class(ci).X) for ci in xrange(d.nclasses)]
    means = [np.atleast_2d(np.mean(d.get_class(ci).X, axis=1)).T 
      for ci in xrange(d.nclasses)]

    # mix global, class and diagonal covariances as specified by a and b:
    a, b = self.alpha, self.beta
    Sg = reduce(operator.add, ([p * c for (p, c) in zip(d.prior, covs)]))
    S0 = np.eye(d.nfeatures) * np.mean(np.diag(Sg))
    covs = [a * S0 + b * Sg + (1. - a - b) * S_i for S_i in covs]

    # calculate and store variables needed for classification
    # (1) log p(x | S, m) = -.5 * (k log(2\pi) + log |sigma|
    #       + (x - m)^T S^{-1} (x - m))
    k = d.nfeatures
    const = [k * np.log(2 * np.pi) - np.log(np.linalg.det(Si)) for Si in covs]
    S_is = [np.linalg.pinv(S) for S in covs]
    self.means, self.const, self.S_is = means, const, S_is

  def apply_(self, d):
    '''Ouput log(p(x | \Sigma_i, \mu_i))'''
    X = []
    for (m, const, S_inv) in zip(self.means, self.const, self.S_is):
      Xc = (d.X - m)
      logp = -.5 * (np.sum(np.dot(Xc.T, S_inv).T * Xc, axis=0) + const)
      X.append(logp)
    return DataSet(X=X, feat_lab=d.cl_lab, default=d)

  def __str__(self):
    return 'RDA(alpha=%.3f, beta=%.3f)' % (self.alpha, self.beta)

class NMC(RDA):
  def __init__(self, **kwargs): RDA.__init__(self, alpha=1, beta=0, **kwargs)
  def __str__(self): return 'NMC'

class LDA(RDA):
  def __init__(self, **kwargs): RDA.__init__(self, alpha=0, beta=1, **kwargs)
  def __str__(self): return 'LDA'

class QDA(RDA):
  def __init__(self, **kwargs): RDA.__init__(self, alpha=0, beta=0, **kwargs)
  def __str__(self): return 'QDA'
