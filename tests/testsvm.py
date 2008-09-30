import unittest
import numpy as np
import numpy.linalg as la

from algorithms.svm import *

class TestSVM(unittest.TestCase):
  def setUp(self):
    pass

  def test_linear(self): 
    '''Test simple linear SVM'''
    # Make data
    data = np.array([[-1, 0, 1], [-1, 1, 1], [0, 0, 1], [0, 1, 1],
                     [1, 0, -1], [1, 1, -1], [2, 0, -1], [2, 1, -1]]
                     ).astype(np.float64)
    xs = data[:, :2]
    ys = data[:, -1].reshape(data.shape[0], 1)

    # Train SVM
    C = 100
    svm = SupportVectorMachine(C=C)
    svm.train(xs, ys)
    
    # Test if the instances are correctly classified
    ys2 = svm.test(xs)[:,0].reshape(xs.shape[0], 1)
    self.assert_((ys2 == ys).all())
    
    # Check if the right Support Vectors are found
    self.assert_((svm.model['SVs'] == xs[2:6]).all())

    # Check if the alphas satisfy the contraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/data.shape[0]).all())

    # Test b in f(x) = ax + b
    self.assert_((svm.test(np.array([[.5, 0], [.5, 1]])) == 
      np.zeros((2, 1))).all())
    self.assert_((svm.test(xs[2:6])[:, 0].flatten() == 
      np.array([1., 1, -1, -1])).all())
  
  def test_nonlinear(self): 
    '''Test simple RBF SVM on a XOR problem'''
    # Make data
    data = np.array([[0, 0, 1], [0, 1, -1], [1, 0, -1], 
      [1, 1, 1]]).astype(np.float64)
    xs = data[:, :2]
    ys = data[:, -1].reshape(data.shape[0], 1)

    # Train SVM
    C = 100
    svm = SupportVectorMachine(C=C, kernel='rbf', sigma=0.5)
    svm.train(xs, ys)

    # Test if the instances are correctly classified
    ys2 = svm.test(xs)[:,0].reshape(xs.shape[0], 1)
    self.assert_((ys2 == ys).all())
    
    # Check if all instances are support vectors
    self.assert_(len(svm.model['SVs']) == 4)

    # Check if the alphas satisfy the contraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/xs.shape[0]).all())

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestSVM))
  return suite

if __name__ == '__main__':
  unittest.main()
