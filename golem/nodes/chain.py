import operator
import numpy as np
from ..cv import strat_splits
from basenode import BaseNode

class Chain(BaseNode):
  def __init__(self, nodes):
    BaseNode.__init__(self)
    self.nodes = list(nodes)
      
  def train_(self, d):
    for (i, n) in enumerate(self.nodes[:-1]):
      self.log.info('Training %s...' % str(n))
      n.train(d)
      d = n.test(d)
    self.nodes[-1].train(d)
    
  def test_(self, d):
    for n in self.nodes:
      self.log.info('Testing with %s...' % str(n))
      d = n.test(d)
    return d

  def __str__(self):
    return 'Chain (%s)' % ' ->\n'.join([str(n) for n in self.nodes])
