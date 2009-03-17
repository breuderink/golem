import unittest
import os.path
import numpy as np
from .. import data, plots

class TestGaussianData(unittest.TestCase):
  def setUp(self):
    np.random.seed(1) # use same seed to make this test reproducible
    self.d = data.gaussian_dataset([200, 200, 50])

  def test_scatterplot(self):
    '''Test Gaussian distribution by writing a scatterplot to a file'''
    plots.scatter_plot(self.d, 'test_gaussian_4_classes.eps')
  
  def test_ninstances(self):
    d = self.d
    self.assert_(d.ninstances_per_class == [200, 200, 50])
  
  def test_ids(self):
    d = self.d
    self.assert_((d.ids == np.arange(450).reshape(-1, 1)).all())
 
  def test_means_cov(self):
    '''Test if the means and covariance of gaussian_data are correct.'''
    mus = [[0, 0], [2, 1], [5, 6]]
    sigmas = [\
      [[1, 2], [2, 5]],
      [[1, 2], [2, 5]],
      [[1, -1], [-1, 2]]]

    d = self.d
    for ci in range(d.nclasses):
      xs = d.get_class(ci).xs
      mean_diff = np.mean(xs, axis=0) - mus[ci]
      self.assert_((abs(mean_diff) < 0.5).all())
      cov_diff = np.cov(xs, rowvar=0) - sigmas[ci]
      self.assert_((abs(cov_diff) < 0.5).all())
