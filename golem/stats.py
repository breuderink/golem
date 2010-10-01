import numpy as np

def holms_proc(ps, alpha=0.05):
  ps = np.atleast_1d(ps)
  alphas = np.array([bonferroni(c, alpha) for c in range(ps.size, 0, -1)])
  h = ps.copy()
  h[np.argsort(ps)] /= alphas
  return h < 1

def bonferroni(c, alpha=0.05):
  return alpha / c

def sidak(c, alpha=0.05):
  return 1 - (1. - alpha) ** (1./c)
