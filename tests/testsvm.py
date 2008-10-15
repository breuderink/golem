import unittest
import os
import random
import numpy as np
import numpy.linalg as la
import pylab

from algorithms import SupportVectorMachine
from helpers import *
from helpers.plots import *
from dataset import *
import loss

EPSILON = 1e-8

class TestSVM(unittest.TestCase):
  def setUp(self):
    pass

  def test_linear(self): 
    '''Test simple linear SVM'''
    # Make data
    xs = np.array([[-1, 0], [-1, 1], [0, 0], [0, 1], 
      [1, 0], [1, 1], [2, 0], [2, 1]]).astype(np.float64)
    ys = np.array([[1, 0], [1, 0], [1, 0], [1, 0], 
      [0, 1], [0, 1], [0, 1], [0, 1]])
    d = DataSet(xs, ys)

    # Train SVM
    C = 100
    svm = SupportVectorMachine(C=C)
    svm.train(d)
    
    # Test if the instances are correctly classified
    self.assert_(loss.accuracy(svm.test(d)) == 1)
    
    # Check if the right Support Vectors are found
    self.assert_((svm.model['SVs'] == xs[2:6]).all())

    # Check if the alphas satisfy the constraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/xs.shape[0]).all())

    # Test b in f(x) = ax + b
    svm.sign_output = False
    hyperplane_d = DataSet(np.array([[.5, 0], [.5, 1]]), np.zeros((2, 2)))
    self.assert_((svm.test(hyperplane_d).xs == hyperplane_d.ys).all())

    sv_d = d[2:6]
    self.assert_(((svm.test(sv_d).xs - sv_d.ys) < EPSILON).all())
  
  def test_nonlinear(self): 
    '''Test simple RBF SVM on a XOR problem'''
    # Make data
    xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]).astype(np.float64)
    ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    d = DataSet(xs, ys)

    # Train SVM
    C = 100
    svm = SupportVectorMachine(C=C, kernel='rbf', sigma=0.5)
    svm.train(d)

    # Test if the instances are correctly classified
    self.assert_(loss.accuracy(svm.test(d)) == 1)
    
    # Check if all instances are support vectors
    self.assert_(len(svm.model['SVs']) == 4)

    # Check if the alphas satisfy the contraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/xs.shape[0]).all())

class TestSVMPlot(unittest.TestCase):
  def test_svm_plot(self):
    '''Create hyperplane plot for SVM'''
    random.seed(1) # use same seed to make this test reproducible
    d = artificialdata.gaussian_dataset([50, 50])

    svm = SupportVectorMachine(C=1e2, kernel='rbf', sigma=1.5, 
      sign_output=False)
    svm.train(d)

    # Plot SVs and scatter
    SVs = svm.model['SVs']
    pylab.scatter(SVs[:,0], SVs[:,1], s=70, c='r', label='SVs')
    scatter_plot(d)
    plot_classifier_hyperplane(svm, heat_map=True, heat_map_alpha=0.9, 
      fname=os.path.join('tests', 'plots', 'test_nonlinear_svm.eps'))

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestSVM))
  suite.addTest(unittest.makeSuite(TestSVMPlot))
  return suite

if __name__ == '__main__':
  unittest.main()
