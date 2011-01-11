import logging
import numpy as np

from ..dataset import DataSet
from basenode import BaseNode

class LSReg(BaseNode):
  def train_(self, d):
    # Add 'offset-feature'
    X = np.vstack([d.X, np.ones(d.ninstances)])
    # Train least-squares
    self.W, self.residual, _, _ = np.linalg.lstsq(X.T, d.Y.T)
    self.W = self.W.T
    self.mse = self.residual / d.ninstances
    self.log.info('MSE = %s' % self.mse)
    self.cl_lab = d.cl_lab

  def apply_(self, d):
    X = np.vstack([d.X, np.ones(d.ninstances)])
    X = np.dot(self.W, X)
    return DataSet(X=X, feat_lab=self.cl_lab, default=d)

