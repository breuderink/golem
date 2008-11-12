import logging
import numpy as np

from golem import DataSet

log = logging.getLogger('LSReg')
class LSReg:
  def train(self, d):
    # Add 'offset-feature'
    xs = np.hstack([d.xs, np.ones((d.ninstances, 1))])
    # Train least-squares
    self.W, self.residual, rank, s = np.linalg.lstsq(xs, d.ys)
    self.mse = self.residual / d.ninstances
    log.info('MSE = %s' % self.mse)
    self.cl_lab = d.cl_lab

  def test(self, d):
    xs = np.hstack([d.xs, np.ones((d.ninstances, 1))])
    xs = np.dot(xs, self.W)
    return DataSet(xs, feat_lab=self.cl_lab, default=d)

  def __str__(self):
    return 'LSReg'
