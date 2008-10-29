import numpy as np

from svm import SVM
from one_vs_rest import OneVsRest
from one_vs_one import OneVsOne
from model_select import ModelSelect
from pca import PCA
from lsreg import LSReg

from golem import DataSet
class Chain:
  def __init__(self, nodes):
    assert(isinstance(nodes, list))
    self.nodes = nodes
      
  def train(self, d):
    for n in self.nodes:
      n.train(d)
      d = n.test(d)
    
  def test(self, d):
    for n in self.nodes:
      d = n.test(d)
    return d

  def __str__(self):
    return 'Chain (%s)' % ' -> '.join([str(n) for n in self.nodes])

class ZScore:
  def __init__(self):
    self.mean = self.std = None
  def train(self, d):
    self.mean = np.mean(d.xs, axis=0)
    self.std = np.std(d.xs, axis=0)

  def test(self, d):
    zxs = (d.xs - self.mean) / self.std # Careful, uses broadcasting!
    return DataSet(xs=zxs, ys=d.ys, ids=d.ids, class_labels=d.class_labels)

  def __str__(self):
    if self.mean <> None:
      return 'ZScore (mean=%s, std=%s)' % (self.mean, self.std)
    else:
      return 'ZScore (untrained)'

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
    return DataSet(xs, d.ys, d.ids, class_labels=d.class_labels)

  def __str__(self): return 'RandomClassifier ()'

