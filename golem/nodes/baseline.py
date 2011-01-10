import numpy as np
from ..dataset import DataSet
from basenode import BaseNode

class PriorClassifier(BaseNode):
  def train_(self, d):
    self.mfc = np.argmax(d.ninstances_per_class)

  def apply_(self, d):
    X = np.zeros(d.Y.shape)
    X[self.mfc] = 1
    return DataSet(X=X, default=d)

  def __str__(self): 
    if hasattr(self, 'mfc'):
      return 'PriorClassifier (class=%d)' % self.mfc
    else:
      return 'PriorClassifier'

class RandomClassifier(BaseNode):
  def apply_(self, d):
    return DataSet(X=np.random.random(d.Y.shape), default=d)

class WeakClassifier(BaseNode):
  '''
  This is a simulation of a weak classifier. It generates random output,
  with a slight bias towards the true labels. 
  *THE TRUE LABELS ARE USED IN THE TEST METHOD*.
  '''
  def apply_(self, d):
    return DataSet(X=np.random.random(d.Y.shape) + .10 * d.Y, default=d)
