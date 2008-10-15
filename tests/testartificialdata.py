import unittest
import os.path
import numpy
from numpy import random
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
    '''Test if the means and covariance of gaussian_data are correct.'''
    mus = [[0, 0], [2, 1], [5, 6]]
    sigmas = [\
      [[1, 2], [2, 5]],
      [[1, 2], [2, 5]],
      [[1, -1], [-1, 2]]]

    d = self.d
    for ci in range(d.nclasses):
      indices = d.ys[:, ci] == 1
      xs = d.xs[indices, :]
      mean_diff = numpy.mean(xs, axis=0) - mus[ci]
      self.assert_((abs(mean_diff) < 0.5).all())
      cov_diff = numpy.cov(xs, rowvar=0) - sigmas[ci]
      self.assert_((abs(cov_diff) < 0.5).all())
