import numpy as np
from scipy import stats
from .. import DataSet

class RDA:
  def __init__(self, alpha=.3, beta=.3):
    '''
    Regularized Discriminant Analysis, Alpaydin, p.98, Eq. 5.29:
    S_i^{'} = \alpha \sigma^2I + \beta S + (1 - \alpha - \beta)S_i
    
    alpha = beta = 0 results in a quadratic classfier,
    alpha = 0, beta = 1 results in a linear classifier,
    alpha = 1, beta = 0 results in a nearest mean classifier.
    '''
    self.alpha = float(alpha)
    self.beta = float(beta)

  def train(self, d):
    self.means = means = []
    covs = []
    self.priors = np.asarray(d.ninstances_per_class) / float(d.ninstances)
    for ci in range(d.nclasses):
      cd = d.get_class(ci)
      means.append(np.mean(cd.xs, axis=0))
      covs.append(np.cov(cd.xs, rowvar=False))

    a, b = self.alpha, self.beta
    ss = np.var(d.xs)
    S = np.cov(d.xs, rowvar=False)
    self.covs = [a * ss * np.eye(d.nfeatures) + b * S + 
      (1 - a - b) * Si for Si in covs]

  def test(self, d):
    '''Ouput log(p(x | class_i))'''
    xs = []
    for (ci, (m, S, P)) in enumerate(zip(self.means, self.covs, self.priors)):
      # (1) Alpaydin, p.93, Eq 5.20:
      # g_i(x) = x^TW_ix + w_i^T x + w_{i0}
      # where: 
      # W_i = -\frac{1}{2}S_i^{-1}
      # w_i = S_i^{-1}m_i
      # w_{i0} = -\frac{1}{2}m_i^TS_i^{-1}m_i - 
      #   \frac{1}{2} log \left| S_i \right| + log\^P(C_i)
      S_inv = np.linalg.pinv(S)
      Wi = -0.5 * S_inv
      wi = np.dot(S_inv, m)
      wi0 = -0.5 * np.dot(np.dot(m.T, S_inv), m) - \
        0.5 * np.log(np.linalg.det(S)) + np.log(P)

      # Vectorized variant of (1):
      gi = np.sum(np.dot(d.xs, Wi) * d.xs, axis=1)  + np.dot(wi.T, d.xs.T) + wi0
      xs.append(gi.reshape(-1, 1))

    xs = np.hstack(xs)
    return DataSet(xs=xs, default=d)
      
