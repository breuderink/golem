import numpy as np
from .. import DataSet

class FeatMap:
  def __init__(self, mapping):
    self.mapping = mapping

  def train(self, d):
    pass

  def test(self, d):
    xs = np.asarray(map(self.mapping, d.nd_xs))
    return DataSet(xs=xs.reshape(d.ninstances, -1), feat_shape=xs.shape[1:],
      default=d)
