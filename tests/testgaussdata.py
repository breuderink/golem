import unittest
import os.path
import scipy
from scipy import random
from helpers import *

class TestGaussianData(unittest.TestCase):
  def setUp(self):
    random.seed(1) # use same seed to make this test reproducible
    self.d = artificialdata.gaussian_dataset([200, 200, 50])

  def test_scatterplot(self):
    '''Test Gaussian distribution by writing a scatterplot to a file'''
    plots.scatter_plot(self.d, os.path.join('tests', 'plots',\
      'test_gaussian_3_classes.eps'))
  
  def test_means_cov(self):
    '''Test if the means and covariance of GaussianData are correct.'''
    mus = [\
      [0, 0], 
      [2, 1],
      [5, 6]]
    sigmas = [\
      [[1, 2], [2, 5]],
      [[1, 2], [2, 5]],
      [[1, -1], [-1, 2]]]

    d = self.d
    xs = d.get_xs()
    ys = d.get_ys()
    for cli in range(len(d.labels)):
      cl_instances = scipy.array([xs[i] for i in range(d.ninstances)\
        if ys[i] == d.labels[cli]])
      mean_diff = scipy.mean(cl_instances, axis=0) - mus[cli]
      self.assert_((abs(mean_diff) < 0.5).all())
      cov_diff = scipy.cov(cl_instances, rowvar=0) - sigmas[cli]
      self.assert_((abs(cov_diff) < 0.5).all())

def suite():
  suite = unittest.TestSuite()
  suite.addTest(unittest.makeSuite(TestGaussianData))
  return suite
