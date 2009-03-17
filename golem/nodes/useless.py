import numpy as np
from ..dataset import DataSet

class PriorClassifier:
  def __init__(self):
    self.mfc = None

  def train(self, d):
    self.nclasses = d.nclasses
    self.mfc = np.argmax(d.ninstances_per_class)

  def test(self, d):
    xs = np.zeros((d.ninstances, self.nclasses))
    xs[:, self.mfc] = 1
    return DataSet(xs, d.ys, d.ids, class_labels=d.class_labels)

  def __str__(self): 
    if self.mfc <> None:
      return 'PriorClassifier (class=%d)' % self.mfc
    else:
      return 'PriorClassifier ()'

class RandomClassifier:
  def train(self, d):
    self.nclasses = d.nclasses

  def test(self, d):
    xs = np.random.random((d.ninstances, self.nclasses))
    return DataSet(xs, d.ys, d.ids, cl_lab=d.cl_lab)

  def __str__(self): return 'RandomClassifier ()'
