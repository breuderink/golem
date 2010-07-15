import math
import numpy as np

def build_kernel_matrix(xs_row, xs_col, kernel=None, **params):
  '''Build Gramm-matrix or kernel-matrix'''
  if kernel == None or kernel == 'linear':
    kernel_matrix = build_kernel_matrix(xs_row, xs_col, 'poly', degree=1)
  elif kernel=='poly':
    d = params['degree']
    assert isinstance(d, int) and d > 0
    kernel_matrix = np.dot(xs_row, xs_col.T) ** d
  elif kernel=='rbf':
    sigma = float(params['sigma'])
    assert sigma > 0
    # calculate k(a, b) = \exp(-\frac{||a-b||^2}{(2 * \sigma^2)}
    # ||a-b||^2 = \sum(a_i-b_i)^2 = a \cdot a + b \cdot b - 2 (a \cdot b)
    rrdot = np.sum(xs_row ** 2, axis=1).reshape(-1, 1)
    ccdot = np.sum(xs_col ** 2, axis=1).reshape(1, -1)
    rcdot = np.dot(xs_row, xs_col.T)
    kernel_matrix = -2 * rcdot + rrdot + ccdot
    return np.exp(-kernel_matrix/(2 * sigma ** 2))
  else:
    # Manually fill kernel matrix with kernel function
    nrows, ncols = xs_row.shape[0], xs_col.shape[0]
    kernel_matrix = np.zeros((nrows, ncols))
    for r in xrange(nrows):
      for c in xrange(ncols):
        kernel_matrix[r, c] = kernel(xs_row[r, :], xs_col[c, :])
  assert kernel_matrix.dtype in (np.float32, np.float64)
  assert np.all(np.isfinite(kernel_matrix))
  return kernel_matrix
