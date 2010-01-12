import unittest, os.path
import numpy as np
import matplotlib.pyplot as plt

from .. import DataSet, loss, data, plots, helpers
from ..nodes import SVM


class TestSVM(unittest.TestCase):
  def test_linear(self): 
    '''Test simple linear SVM'''
    # Make data
    xs = np.array([[-1, 0], [-1, 1], [0, 0], [0, 1], 
      [1, 0], [1, 1], [2, 0], [2, 1]]).astype(np.float64)
    ys = np.array([[1, 0], [1, 0], [1, 0], [1, 0], 
      [0, 1], [0, 1], [0, 1], [0, 1]])
    d = DataSet(xs, ys, None)

    # Train SVM
    C = 100
    svm = SVM(C=C)
    svm.train(d)
    
    # Test if the instances are correctly classified
    self.assertEqual(loss.accuracy(svm.test(d)), 1)
    
    # Check if the right Support Vectors are found
    np.testing.assert_equal(svm.model['SVs'], xs[2:6])

    # Check if the alphas satisfy the constraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/xs.shape[0]).all())

    # Test b in f(x) = ax + b
    hyperplane_d = DataSet(xs=np.array([[.5, 0], [.5, 1]]), ys=np.zeros((2, 2)))
    np.testing.assert_almost_equal(svm.test(hyperplane_d).xs, hyperplane_d.ys)

    # Test SVs
    sv_d = d[2:6]
    np.testing.assert_almost_equal(svm.test(sv_d).xs, sv_d.ys * 2. - 1)
  
  def test_nonlinear(self): 
    '''Test simple RBF SVM on a XOR problem'''
    # Make data
    xs = np.array([[0, 0], [1, 0], [0, 1], [1, 1.]])
    ys = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    d = DataSet(xs=xs, ys=ys)

    # Train SVM
    C = 100
    svm = SVM(C=C, kernel='rbf', sigma=0.5)
    svm.train(d)

    # Test if the instances are correctly classified
    self.assertEqual(loss.accuracy(svm.test(d)), 1)
    
    # Check if all instances are support vectors
    self.assertEqual(len(svm.model['SVs']), 4)

    # Check if the alphas satisfy the contraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.model['alphas']).all())
    self.assert_((svm.model['alphas'] < C/xs.shape[0]).all())

class TestSVMPlot(unittest.TestCase):
  def test_svm_plot(self):
    '''Create hyperplane plot for SVM'''
    np.random.seed(0) # use same seed to make this test reproducible
    d = data.gaussian_dataset([50, 50])

    svm = SVM(C=1e2, kernel='rbf', sigma=1.5)
    svm.train(d)
    self.assertEqual(len(svm.model['SVs']), 49)

    # Plot SVs and scatter
    SVs = svm.model['SVs']
    plt.clf()
    plt.scatter(SVs[:,0], SVs[:,1], s=70, c='r', label='SVs')
    plots.plot_classifier(svm, d, densities=True)
    plt.savefig(os.path.join('out', 'test_nonlinear_svm.eps'))
