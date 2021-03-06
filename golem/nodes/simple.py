import numpy as np
from basenode import BaseNode
from .. import DataSet

class ApplyOverInstances(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    instances = np.asarray(map(self.mapping, np.rollaxis(d.ndX, -1)))
    ndX = np.rollaxis(instances, 0, len(instances.shape))
    return DataSet(X=ndX.reshape(-1, d.ninstances), 
      feat_shape=ndX.shape[:-1], default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)

class FeatMap(ApplyOverInstances): pass

class ApplyOverFeats(BaseNode):
  def __init__(self, mapping):
    BaseNode.__init__(self)
    self.mapping = mapping

  def apply_(self, d):
    X = np.apply_along_axis(self.mapping, 0, d.X)
    return DataSet(X=X, default=d)

  def __str__(self):
    return '%s (with mapping "%s")' % (self.name, self.mapping.__name__)


class ZScore(BaseNode):
  def train_(self, d):
    self.mean = np.atleast_2d(np.mean(d.X, axis=1)).T
    self.std = np.atleast_2d(np.std(d.X, axis=1)).T

  def apply_(self, d):
    return DataSet(X=(d.X - self.mean) / self.std, default=d)
