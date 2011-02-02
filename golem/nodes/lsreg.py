import logging
import numpy as np

from ..dataset import DataSet
from basenode import BaseNode

class LSReg(BaseNode):
  """  
  Least squares regression node to fit a line on the data set.
  
  train() determines optimal line through the training data d.
    d.ys = m . d.xs + c
    
  apply() determines the estimated labels given d.xs.
  
  
  Further reading:
  
  [1] Numpy documentation for the least squares regression function
  http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    
  [2] Wikipedia: Ordinary (linear) least squares
  http://en.wikipedia.org/wiki/Ordinary_least_squares
  """
  
  def train_(self, d):
    """
    Fits a straight line through the dataset (xs, ys), resulting
    in the least-squares solution, minimizing the Euclidean
    distance from the data points to the estimated line.
    
    Raises: LinAlgError if the least squares regression 
    does not converge.    
    """
    # Add 'offset-feature'
    xs = np.hstack([d.xs, np.ones((d.ninstances, 1))])
    # Train least-squares
    self.W, self.residual, rank, s = np.linalg.lstsq(xs, d.ys)
    self.mse = self.residual / d.ninstances
    self.log.info('MSE = %s' % self.mse)
    self.cl_lab = d.cl_lab

  def apply_(self, d):
    """
    Returns a dataset with the (through least squares regression) estimated 
    label values given the sample values.
    
    @@TODO should it not by xs.W + residual ? Or is line through 0 assumed?
    """
    xs = np.hstack([d.xs, np.ones((d.ninstances, 1))])
    xs = np.dot(xs, self.W)
    return DataSet(xs, feat_lab=self.cl_lab, default=d)

