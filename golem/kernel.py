import math
import numpy as np

def build_kernel_matrix(X_row, X_col, kernel=None, **params):
  '''Build Gramm-matrix or kernel-matrix'''
  if kernel == None or kernel == 'linear':
    kernel_matrix = np.dot(X_row.T, X_col)
  elif kernel=='poly':
    d = params['degree']
    assert isinstance(d, int) and d > 0
    kernel_matrix = np.dot(X_row.T, X_col) ** d
  elif kernel=='rbf':
    sigma = float(params['sigma'])
    assert sigma > 0
    # calculate k(a, b) = \exp(-\frac{||a-b||^2}{(2 * \sigma^2)}
    # ||a-b||^2 = \sum(a_i-b_i)^2 = a \cdot a + b \cdot b - 2 (a \cdot b)
    rrdot = np.sum(X_row ** 2, axis=0).reshape(-1, 1)
    ccdot = np.sum(X_col ** 2, axis=0).reshape(1, -1)
    rcdot = np.dot(X_row.T, X_col)
    kernel_matrix = -2 * rcdot + rrdot + ccdot
    return np.exp(-kernel_matrix/(2 * sigma ** 2))
  else:
    # Manually fill kernel matrix with kernel function
    nrows, ncols = X_row.shape[1], X_col.shape[1]
    kernel_matrix = np.zeros((nrows, ncols)) * np.nan
    for r in xrange(nrows):
      for c in xrange(ncols):
        kernel_matrix[r, c] = kernel(X_row[:,r], X_col[:,c])

  assert kernel_matrix.dtype in (np.float32, np.float64)
  assert np.all(np.isfinite(kernel_matrix))
  return kernel_matrix
