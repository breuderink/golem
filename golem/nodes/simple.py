import numpy as np
from basenode import BaseNode
from .. import DataSet

class FeatMap(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    xs = np.asarray(map(self.mapping, d.nd_xs))
    return DataSet(xs=xs.reshape(d.ninstances, -1), feat_shape=xs.shape[1:],
      default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

class ZScore(BaseNode):
  def train_(self, d):
    self.mean = np.mean(d.xs, axis=0)
    self.std = np.std(d.xs, axis=0)

  def apply_(self, d):
    return DataSet(xs=(d.xs - self.mean) / self.std, default=d)
