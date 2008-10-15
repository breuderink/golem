import unittest
import numpy as np
import numpy.linalg as la

from algorithms.kernel import *

def test_kernel_function(kernel_matrix, kernel, x1, x2):
  '''Test if kernel-matrix is built with the correct kernel function'''
  (m, n) = kernel_matrix.shape
  for r in range(m):
    for c in range(n):
      if kernel(x1[r,:], x2[c, :]) <> kernel_matrix[r, c]:
        print kernel(x1[r,:], x2[c, :]) , '<> ', kernel_matrix[r, c]
        return False
  return True

def test_kernel_props(kernel_matrix):
  '''Test positive trace and symmetry of kernel matrix'''
  if not (kernel_matrix == kernel_matrix.T).all():
    return False
  if np.trace(kernel_matrix) < 0: 
    return False
  return True

class TestKernel(unittest.TestCase):
  def setUp(self):
    self.x1 = np.array(([0, 0], [1, 0], [-4, 4])).astype(np.float64);
    self.x2 = np.array(([-1.5, -5], [1e-5, 1e5]));

  def test_int_features(self): 
    '''Verify that int-featues are not accepted by build_kernel_matrix.'''
    x1_bad = np.array(([0, 0], [1, 0], [-4, 4]));
    self.assertRaises(Exception, build_kernel_matrix, x1_bad, x1_bad)

  def test_linear(self):
    '''Test linear kernel'''
    x1, x2 = self.x1, self.x2
    kernel = lambda a, b: np.dot(a, b)
    k12 = build_kernel_matrix(x1, x2)
    k11 = build_kernel_matrix(x1, x1)
    
    self.assert_(test_kernel_props(k11))
    self.assert_(test_kernel_function(k11, kernel, x1, x1))
    self.assert_(test_kernel_function(k12, kernel, x1, x2))

  def test_rbf(self):
    '''Test rbf kernel'''
    x1, x2 = self.x1, self.x2
    def kernel(a, b, sigma):
      d = a - b
      return math.exp(-(np.dot(d, d))/(2. * sigma ** 2))

    # Test different kernel sizes
    for s in [.1, 5, 20]:
      k11 = build_kernel_matrix(x1, x1, 'rbf', sigma = s)
      k12 = build_kernel_matrix(x1, x2, 'rbf', sigma = s)
      self.assert_(k11.dtype == np.float64)
      self.assert_(k12.dtype == np.float64)
      
      self.assert_(test_kernel_props(k11))
      self.assert_(test_kernel_function(k11, 
        lambda a, b: kernel(a, b, s), x1, x1))
      self.assert_(test_kernel_function(k12, 
        lambda a, b: kernel(a, b, s), x1, x2))

    # Test that this kernel fails for sigma == 0
    self.assertRaises(Exception, build_kernel_matrix, x1, x1, 'rbf', sigma = 0)
  
  def test_poly(self):
    '''Test polynomial kernel'''
    x1, x2 = self.x1, self.x2
    def kernel(a, b, degree):
      return np.dot(a, b) ** float(degree)

    # Test different kernel sizes
    for d in [1, 2, 3, 20]:
      k11 = build_kernel_matrix(x1, x1, 'poly', degree = d)
      k12 = build_kernel_matrix(x1, x2, 'poly', degree = d)
      
      self.assert_(k11.dtype == np.float64)
      self.assert_(k12.dtype == np.float64)
      
      self.assert_(test_kernel_props(k11))
      self.assert_(test_kernel_function(k11, 
        lambda a, b: kernel(a, b, d), x1, x1))
      self.assert_(test_kernel_function(k12, 
        lambda a, b: kernel(a, b, d), x1, x2))

    # Test that this kernel failes for invalid degrees
    self.assertRaises(Exception, build_kernel_matrix, 
      x1, x1, 'poly', degree = 0)
    self.assertRaises(Exception, build_kernel_matrix, 
      x1, x1, 'poly', degree = 1.5)

if __name__ == '__main__':
  unittest.main()
