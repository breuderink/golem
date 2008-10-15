__all__ = ['kernel', 'svm', 'one_vs_rest']

import numpy as np
from svm import SupportVectorMachine
from one_vs_rest import OneVsRest
from dataset import *

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

class ZScore:
  def train(self, d):
    self.mean = np.mean(d.xs, axis=0)
    self.std = np.std(d.xs, axis=0)

  def test(self, d):
    zxs = (d.xs - self.mean) / self.std # Careful, uses broadcasting!
    return DataSet(xs=zxs, ys=d.ys, ids=d.ids, class_labels=d.class_labels)

  def __str__(self):
    return 'ZScore (mean=%s, std=%s)' % (self.mean, self.std)


