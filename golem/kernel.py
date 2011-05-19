import numpy as np

def build_kernel_matrix(X_row, X_col, kernel=None, **params):
  '''Build Gramm-matrix or kernel-matrix'''
  X_row, X_col = np.atleast_2d(X_row, X_col)
  if kernel == None or kernel == 'linear':
    kernel_matrix = np.dot(X_row.T, X_col)
  elif kernel=='poly':
    d = params['degree']
    offset = params.get('offset', 1.)
    scale = params.get('scale', 1.)
    assert isinstance(d, int) and d > 0
    assert isinstance(offset, float) and offset >= 0
    assert isinstance(scale, float) and scale > 0
    kernel_matrix = (scale * np.dot(X_row.T, X_col) + offset) ** d
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
    # manually fill kernel matrix with kernel function
    nrows, ncols = X_row.shape[1], X_col.shape[1]
    kernel_matrix = np.zeros((nrows, ncols)) * np.nan
    for r in xrange(nrows):
      for c in xrange(ncols):
        kernel_matrix[r, c] = kernel(X_row[:,r], X_col[:,c])

  # sanity check computed kernel matrix
  assert kernel_matrix.dtype in (np.float32, np.float64)
  assert np.all(np.isfinite(kernel_matrix))
  return kernel_matrix

def kernel_cv_fold(K, folds, fi):
  '''
  Return kernel for training and kernel for testing for fold fi:
  >>> K = np.arange(5*5).reshape(5, 5)
  >>> K
  array([[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24]])

  >>> K_tr, K_te = kernel_cv_fold(K, [0, 0, 1, 1, 2], 1)
  >>> K_tr
  array([[ 0,  1,  4],
         [ 5,  6,  9],
         [20, 21, 24]])
  >>> K_te
  array([[ 2,  3],
         [ 7,  8],
         [22, 23]])
  '''
  folds = np.atleast_1d(folds)
  tr = K[folds!=fi][:, folds!=fi]
  te = K[folds!=fi][:, folds==fi]
  return tr, te
