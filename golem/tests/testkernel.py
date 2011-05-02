import unittest, math
import numpy as np
import numpy.linalg as la

from ..kernel import build_kernel_matrix

def check_kernel_props(kernel_matrix):
  '''Test positive trace and symmetry of kernel matrix'''
  if not np.allclose(kernel_matrix, kernel_matrix.T):
    raise AssertionError('Kernel is not symmetric')
  if np.trace(kernel_matrix) < 0: 
    raise AssertionError('Kernel has a negative trace')
  if not np.all(np.linalg.eigvalsh(kernel_matrix) + 1e-15 >= 0):
    raise AssertionError('Kernel is not positive semidefinite: min eigv.=%g'
      % np.min(np.linalg.eigvalsh(kernel_matrix)))

class TestKernel(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    self.X1 = np.random.randn(100, 10)
    self.X2 = np.random.randn(100, 5)
    np.seterr(over='raise', under='raise')

  def test_complex_features(self): 
    '''
    Verify that complex-features are not (yet) accepted by build_kernel_matrix.
    '''
    X1_bad = self.X1.astype(np.complex)
    self.assertRaises(Exception, build_kernel_matrix, X1_bad, X1_bad)

  def test_custom_kernel(self):
    '''Test user supplied kernel. Used to verify other kernels.'''
    f = lambda a, b: 2 * np.dot(a.T, b)
    X1, X2 = self.X1, self.X2
    k12 = build_kernel_matrix(X1, X2, f)
    np.testing.assert_almost_equal(k12, 2 * build_kernel_matrix(X1, X2))

  def test_linear(self):
    '''Test linear kernel'''
    X1, X2 = self.X1, self.X2
    check_kernel_props(build_kernel_matrix(X1, X1))
    np.testing.assert_equal(
      build_kernel_matrix(X1, X2), np.dot(X1.T, X2))

  def test_rbf(self):
    '''Test RBF kernel'''
    def rbf_kernel(a, b, sigma):
      return math.exp(-(np.dot(a-b, a-b))/(2. * sigma ** 2))
    X1, X2 = self.X1, self.X2

    # test different kernel sizes
    for s in [.1, 5, 20]:
      check_kernel_props(build_kernel_matrix(X1, X1, 'rbf', sigma=s))
      np.testing.assert_almost_equal(
        build_kernel_matrix(X1, X2, 'rbf', sigma=s),
        build_kernel_matrix(X1, X2, lambda a, b: rbf_kernel(a, b, s)))

    # test that this kernel fails for sigma == 0
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'rbf', sigma=0)

  def test_poly(self):
    '''Test polynomial kernel'''
    def poly_kernel(a, b, degree, offset=1., scale=1.):
      return (scale * np.dot(a, b) + offset) ** float(degree)
    X1, X2 = self.X1, self.X2

    for o in [0., 1.]: # test homogeneous and inhomogeneous polynomial kernels
      for s in [.5, 1.]: # test different scales
        for d in [1, 2, 3, 6]: # test different kernel sizes
          check_kernel_props(
            build_kernel_matrix(X1, X1, 'poly', degree=d, scale=s, offset=o))
          np.testing.assert_almost_equal(
            build_kernel_matrix(X1, X2, 'poly', degree=d, scale=s, offset=o), 
            build_kernel_matrix(X1, X2, 
              lambda a, b: poly_kernel(a, b, d, scale=s, offset=o)))

    # Test that this kernel fails for invalid degrees
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'poly', degree=0)
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'poly', degree=-1)
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'poly', degree=.7)
