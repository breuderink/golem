import unittest, os.path
import numpy as np
import matplotlib.pyplot as plt

from .. import DataSet, perf, data, plots, helpers
from ..nodes import SVM

class TestSVM(unittest.TestCase):
  def test_linear(self): 
    '''Test simple linear SVM'''
    # make data
    '''
       0  1  2  3
    0  o  o  +  +
    1  o  o  +  +
    '''
    X = np.array([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]], float)
    Y = helpers.to_one_of_n([0, 0, 0, 0, 1, 1, 1, 1])
    d = DataSet(X=X, Y=Y)

    # train SVM
    c = 1e10
    svm = SVM(c=[c])
    svm.train(d)

    # verify alphas
    alphas = svm.alphas.flatten()
    np.testing.assert_almost_equal(alphas, [0, 0, 1, 1, 1, 1, 0, 0])

    # extract hyperplane using \vec{w} = X diag(\vec{y}) \vec{\alpha}:
    A = [d.X, np.diag(np.where(Y[0] == 1, -1, 1)), alphas.reshape(-1, 1)]
    w = reduce(np.dot, A)
    b = svm.b

    # verify w, b using known values:
    np.testing.assert_almost_equal(w.T, [[2, 0]])
    np.testing.assert_almost_equal(b, -3, decimal=4)
    
    # test classification by SVM itself
    np.testing.assert_almost_equal(svm.apply(d).X, np.dot(
      np.atleast_2d([-1, 1]).T, 
      np.atleast_2d([-3, -3, -1, -1, 1, 1, 3, 3])),
      decimal=4)
    
    # check sparsity of solution
    np.testing.assert_equal(svm.svs.X, d.X[:,2:6])
  
  def test_nonlinear(self): 
    '''Test simple RBF SVM on a XOR problem'''
    # Make data
    X = np.array([[0., 1, 0, 1], [0, 0, 1, 1]])
    Y = helpers.to_one_of_n([0, 1, 0, 1])
    d = DataSet(X=X, Y=Y)

    # Train SVM
    c = 100
    svm = SVM(c=[c], kernel='rbf', sigma=0.5)
    svm.train(d)

    # Test if the instances are correctly classified
    self.assertEqual(perf.accuracy(svm.apply(d)), 1)
    
    # Check if all instances are support vectors
    self.assertEqual(svm.svs, d)

    # Check if the alphas satisfy the constraints
    # 0 < all alphas < C/m where m is the number of instances
    self.assert_((0 < svm.alphas).all())
    self.assert_((svm.alphas < c/d.ninstances).all())

class TestSVMPlot(unittest.TestCase):
  def test_svm_plot(self):
    '''Create hyperplane plot for SVM'''
    np.random.seed(0) # use same seed to make this test reproducible
    d = data.gaussian_dataset([50, 50])

    svm = SVM(c=1e2, kernel='rbf', sigma=1.5)
    svm.train(d)
    self.assertEqual(svm.svs.ninstances, 40)

    # Plot SVs and scatter
    SVs = svm.svs.X
    plt.clf()
    plt.scatter(SVs[0], SVs[1], s=70, c='r', label='SVs')
    plots.plot_classifier(svm, d, densities=True)
    plt.savefig(os.path.join('out', 'test_nonlinear_svm.png'))
