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
  return True

class TestKernel(unittest.TestCase):
  def setUp(self):
    self.x1 = np.vstack([[[0., 0], [1, 0], [-4, 4], [1e-5, 1e5]], 
      np.random.rand(10, 2)])
    self.x2 = np.vstack([[[-1.5, -5], [1e-5, 1e5]], np.random.rand(5, 2)])

  def test_complex_features(self): 
    '''Verify that complex-featues are not accepted by build_kernel_matrix.'''
    x1_bad = np.array(([0, 0], [1, 0], [-4, 4])).astype(complex)
    self.assertRaises(Exception, build_kernel_matrix, x1_bad, x1_bad)

  def test_custom_kernel(self):
    '''Test user supplied kernel. Used to verify other kernels.'''
    f = lambda a, b: np.dot(a * 1e3, b)
    x1, x2 = self.x1.astype(np.float32), self.x2.astype(np.float32)
    k12 = build_kernel_matrix(x1, x2, f)
    self.assertEqual(k12.shape, (x1.shape[0], x2.shape[0]))

    for i in range(k12.shape[0]):
      for j in range(k12.shape[1]):
        self.assertAlmostEqual(k12[i,j], f(x1[i], x2[j]))

  def test_linear(self):
    '''Test linear kernel'''
    x1, x2 = self.x1, self.x2
    test_kernel_props(build_kernel_matrix(x1, x1))
    self.assert_(np.allclose(build_kernel_matrix(x1, x2), 
      build_kernel_matrix(x1, x2, lambda a, b: np.dot(a, b))))

  def test_rbf(self):
    '''Test rbf kernel'''
    def kernel(a, b, sigma):
      d = a - b
      return math.exp(-(np.dot(d, d))/(2. * sigma ** 2))
    x1, x2 = self.x1, self.x2

    # Test different kernel sizes
    for s in [.1, 5, 20]:
      test_kernel_props(build_kernel_matrix(x1, x1, 'rbf', sigma=s))
      self.assert_(np.allclose(build_kernel_matrix(x1, x2, 'rbf', sigma=s),
        build_kernel_matrix(x1, x2, lambda a, b: kernel(a, b, s))))

    # Test that this kernel fails for sigma == 0
    self.assertRaises(Exception, build_kernel_matrix, x1, x1, 'rbf', sigma=0)

  def test_poly(self):
    '''Test polynomial kernel'''
    def kernel(a, b, degree):
      return np.dot(a, b) ** float(degree)
    x1, x2 = self.x1, self.x2

    # Test different kernel sizes
    for d in [1, 2, 3, 6]:
      test_kernel_props(build_kernel_matrix(x1, x1, 'poly', degree=d))
      self.assert_(np.allclose(build_kernel_matrix(x1, x2, 'poly', degree=d), 
        build_kernel_matrix(x1, x2, lambda a, b: kernel(a, b, d))))

    # Test that this kernel failes for invalid degrees
    self.assertRaises(Exception, build_kernel_matrix, x1, x1, 'poly', degree=0)
    self.assertRaises(Exception, build_kernel_matrix, 
      x1, x1, 'poly', degree=1.5)
