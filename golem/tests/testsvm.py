import unittest, os.path
import numpy as np
import matplotlib.pyplot as plt

from .. import DataSet, perf, data, plots, helpers
from ..nodes import SVM

class TestSVM(unittest.TestCase):
  def test_linear(self): 
    '''Test simple linear SVM'''
    # Make data
    X = np.array([[-1., -1, 0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1, 0, 1]])
    Y = helpers.to_one_of_n([0, 0, 0, 0, 1, 1, 1, 1])
    d = DataSet(X=X, Y=Y)

    # Train SVM
    C = 100
    svm = SVM(C=C)
    svm.train(d)
    
    # Test if the instances are correctly classified
    self.assertEqual(perf.accuracy(svm.apply(d)), 1)
    
    # Check if the right Support Vectors are found
    np.testing.assert_equal(svm.svs.X, d.X[:,2:6])

    # Check if the alphas satisfy the constraints
    # 0 <= all alphas <= C/m where m is the number of instances
    self.assert_((0 <= svm.alphas).all())
    self.assert_((svm.alphas <= C/d.ninstances).all())

    # Test b in f(x) = ax + b by classifying points on hyperplane
    hyperplane_d = DataSet(X=np.array([[.5, .5], [0, 1]]), Y=np.zeros((2, 2)))
    np.testing.assert_almost_equal(svm.apply(hyperplane_d).X, hyperplane_d.Y)

    # Test SVs
    sv_d = d[2:6]
    np.testing.assert_almost_equal(svm.apply(sv_d).X, sv_d.Y * 2. - 1)
  
  def test_nonlinear(self): 
    '''Test simple RBF SVM on a XOR problem'''
    # Make data
    X = np.array([[0., 1, 0, 1], [0, 0, 1, 1]])
    Y = helpers.to_one_of_n([0, 1, 0, 1])
    d = DataSet(X=X, Y=Y)

    # Train SVM
    C = 100
    svm = SVM(C=C, kernel='rbf', sigma=0.5)
    svm.train(d)

    # Test if the instances are correctly classified
    self.assertEqual(perf.accuracy(svm.apply(d)), 1)
    
    # Check if all instances are support vectors
    self.assertEqual(svm.svs, d)

    # Check if the alphas satisfy the constraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.alphas).all())
    self.assert_((svm.alphas < C/d.ninstances).all())

class TestSVMPlot(unittest.TestCase):
  def test_svm_plot(self):
    '''Create hyperplane plot for SVM'''
    np.random.seed(0) # use same seed to make this test reproducible
    d = data.gaussian_dataset([50, 50])

    svm = SVM(C=1e2, kernel='rbf', sigma=1.5)
    svm.train(d)
    self.assertEqual(svm.svs.ninstances, 40)

    # Plot SVs and scatter
    SVs = svm.svs.X
    plt.clf()
    plt.scatter(SVs[0], SVs[1], s=70, c='r', label='SVs')
    plots.plot_classifier(svm, d, densities=True)
    plt.savefig(os.path.join('out', 'test_nonlinear_svm.eps'))
