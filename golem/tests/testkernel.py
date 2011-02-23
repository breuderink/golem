import unittest, math
import numpy as np
import numpy.linalg as la

from ..kernel import build_kernel_matrix

def test_kernel_props(kernel_matrix):
  '''Test positive trace and symmetry of kernel matrix'''
  if not np.allclose(kernel_matrix, kernel_matrix.T):
    raise AssertionError('Kernel is not symmetric')
  if np.trace(kernel_matrix) < 0: 
    raise AssertionError('Kernel has a negative trace')
  if np.any(np.linalg.eigvals(kernel_matrix) + 1e8 < 0):
    raise AssertionError('Kernel is not postive semidefinite')

test_kernel_props.__test__ = False # prevent nose from thinking this is a test

class TestKernel(unittest.TestCase):
  def setUp(self):
    self.X1 = np.hstack(
      [[[0, 1, -4, 1e-5], [0, 0, -4, 1e5]], np.random.rand(2, 10)])
    self.X2 = np.hstack([[[-1.5, 1e-5], [-5, 1e5]], np.random.rand(2, 5)])

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
    np.testing.assert_equal(k12, 2 * build_kernel_matrix(X1, X2))

  def test_linear(self):
    '''Test linear kernel'''
    X1, X2 = self.X1, self.X2
    test_kernel_props(build_kernel_matrix(X1, X1))
    np.testing.assert_equal(
      build_kernel_matrix(X1, X2), np.dot(X1.T, X2))

  def test_rbf(self):
    '''Test RBF kernel'''
    def rbf_kernel(a, b, sigma):
      return math.exp(-(np.dot(a-b, a-b))/(2. * sigma ** 2))
    X1, X2 = self.X1, self.X2

    # test different kernel sizes
    for s in [.1, 5, 20]:
      test_kernel_props(build_kernel_matrix(X1, X1, 'rbf', sigma=s))
      np.testing.assert_almost_equal(
        build_kernel_matrix(X1, X2, 'rbf', sigma=s),
        build_kernel_matrix(X1, X2, lambda a, b: rbf_kernel(a, b, s)))

    # test that this kernel fails for sigma == 0
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'rbf', sigma=0)

  def test_poly(self):
    '''Test polynomial kernel'''
    def poly_kernel(a, b, degree):
      return np.dot(a, b) ** float(degree)
    X1, X2 = self.X1, self.X2

    # Test different kernel sizes
    for d in [1, 2, 3, 6]:
      test_kernel_props(build_kernel_matrix(X1, X1, 'poly', degree=d))
      np.testing.assert_almost_equal(
        build_kernel_matrix(X1, X2, 'poly', degree=d), 
        build_kernel_matrix(X1, X2, lambda a, b: poly_kernel(a, b, d)))

    # Test that this kernel fails for invalid degrees
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'poly', degree=0)
    self.assertRaises(Exception, build_kernel_matrix, X1, X1, 'poly', 
      degree=1.5)
