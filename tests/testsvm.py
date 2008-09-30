import unittest
import os
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

class SVMPlot(unittest.TestCase):
  def test_svm_plot(self):
    '''Create hyperplane plot for SVM'''
    # Load Dataset
    data = pickle.load(open('linsep.bin'))
    xs = np.vstack((np.array(data['X']).transpose(), 
      np.array(data['Y']).transpose()))
    ys = np.array([-1. for i in range(data['X'].size[1])] + \
      [1. for i in range(data['Y'].size[1])])
    ys = ys.reshape(ys.size, 1) # make column-vector

    svm = SupportVectorMachine(C=100, kernel='rbf', sigma=2.9, 
      sign_output=False)
    svm.train(xs, ys)

    # add scatter
    SVs = svm.model['SVs']
    pylab.scatter(SVs[:,0], SVs[:,1], s=70, c='w')
    class1 = xs[[i for i in range(len(ys)) if ys[i] == -1], :]
    class2 = xs[[i for i in range(len(ys)) if ys[i] == 1], :]
    pylab.scatter(class1[:, 0], class1[:, 1], c='k', s = 20)
    pylab.scatter(class2[:, 0], class2[:, 1], c='w', s = 20)

    plot_classifier_hyperplane(svm, heat_map_alpha = 0.9, 
      fname=os.path.join('tests', 'plots', 'test_nonlinear_svm.eps'))

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestSVM))
  suite.addTest(unittest.makeSuite(SVMPlot))
  return suite

if __name__ == '__main__':
  unittest.main()
