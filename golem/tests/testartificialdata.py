import unittest
import os.path
import numpy as np
import pylab
from .. import data, plots

class TestGaussianData(unittest.TestCase):
  def setUp(self):
    np.random.seed(1) # use same seed to make this test reproducible
    self.d = data.gaussian_dataset([200, 200, 100])

  def test_scatterplot(self):
    '''Test Gaussian distribution by writing a scatter plot to a file'''
    pylab.clf()
    plots.feat_scatter(self.d)
    pylab.savefig(os.path.join('out', 'test_gaussian_3_classes.eps'))
  
  def test_ninstances(self):
    d = self.d
    self.assert_(d.ninstances_per_class == [200, 200, 100])
  
  def test_means_cov(self):
    '''Test if the means and covariance of Gaussian_data are correct.'''
    mus = [[0, 0], [2, 1], [5, 6]]
    sigmas = [\
      [[1, 2], [2, 5]],
      [[1, 2], [2, 5]],
      [[1, -1], [-1, 2]]]

    d = self.d
    for ci in range(d.nclasses):
      X = d.get_class(ci).X
      np.testing.assert_almost_equal(np.mean(X, axis=1), mus[ci], decimal=0)
      np.testing.assert_almost_equal(np.cov(X), sigmas[ci], decimal=0)

class TestSpirals(unittest.TestCase):
  def setUp(self):
    self.d = data.wieland_spirals()

  def test_scatterplot(self):
    pylab.clf()
    plots.feat_scatter(self.d)
    pylab.savefig(os.path.join('out', 'test_spirals.png'))
