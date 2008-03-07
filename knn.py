# @@TODO the hashing in the classify method is ugly.

from data import Data
from block import Block

def L2(point1, point2):
    return sum([(y-x)**2 for x,y in zip(point1, point2)])**0.5

class KNN(Block):
  def __init__(self, K=1, distf=L2):
    Block.__init__(self)
    self.hyperparams['K'] = K
    self.hyperparams['distf'] = distf.__name__
    self.distf = distf
    self.name = 'KNN'
  
  def train_method(self, data):
    """Train by remembering train samples with labels"""
    assert(len(data.ys[0]) == 1) # lists cannot be hashed, we used one class-label
    self.model.examples = zip(data.xs, data.ys)
    return self.test_method(data)

  def test_method(self, data):
    """Classify an unknown set of samples"""
    return Data([self.classify(x) for x in data.xs], data.ys, data.signature + [self.name])
      
  def classify(self, i):
    """Classify a single instance"""
    distf = self.distf
    dists = [(distf(i, x), y) for (x, y) in self.model.examples]
    neighbours = sorted(dists)[:self.hyperparams['K']]
    
    # count votes
    d = {}
    for (ndist, ny) in neighbours:
      label = ny[0]
      d[label] = d.get(label, 0) + 1
    
    # return marjority vote
    ranked = sorted(d.items(), key=lambda x: x[1])
    return [ranked[0][0]]

