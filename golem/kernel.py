import math
import numpy as np

def build_kernel_matrix(instances_row, instances_col, kernel=None, **params):
  '''Build Gramm-matrix or kernel-matrix'''
  assert(instances_row.dtype == instances_col.dtype == np.float64)
  irow, icol = instances_row, instances_col

  if kernel == None or kernel == 'linear':
    kernel_matrix = build_kernel_matrix(irow, icol, 'poly', degree=1)
  elif kernel=='poly':
    assert params['degree'] > 0
    assert isinstance(params['degree'], int)
    kernel_matrix = np.dot(irow, icol.T) ** params['degree']
  elif kernel=='rbf':
    sigma = float(params['sigma'])
    assert sigma > 0
    # calculate k(a, b) = \exp(-\frac{||a-b||^2}{(2 * \sigma^2)}
    # ||a-b||^2 = \sum(a_i-b_i)^2 = a \cdot a + b \cdot b - 2 (a \cdot b)
    rrdot = np.sum(irow ** 2, axis=1).reshape(-1, 1)
    ccdot = np.sum(icol ** 2, axis=1).reshape(1, -1)
    rcdot = np.dot(irow, icol.T)
    kernel_matrix = -2 * rcdot + rrdot + ccdot
    return np.exp(-kernel_matrix/(2 * sigma ** 2))
  else:
    # Manually fill kernel matrix with kernel function
    nrows, ncols = irow.shape[0], icol.shape[0]
    kernel_matrix = np.zeros((nrows, ncols))
    for r in xrange(nrows):
      for c in xrange(ncols):
        kernel_matrix[r, c] = kernel(irow[r, :], icol[c, :])
  return kernel_matrix
