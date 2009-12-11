import operator
import numpy as np
from ..cv import strat_splits
from basenode import BaseNode

class Chain(BaseNode):
  def __init__(self, nodes):
    BaseNode.__init__(self)
    self.nodes = list(nodes)
      
  def train_(self, d):
    for n in self.nodes:
      self.log.info('Training %s...' % str(n))
      self.log.debug('d = %s' % d)
      n.train(d)
      if n != self.nodes[-1]:
        d = n.test(d)
    
  def test_(self, d):
    for n in self.nodes:
      self.log.info('Testing with %s...' % str(n))
      self.log.debug('d = %s' % d)
      d = n.test(d)
    return d

  def __str__(self):
    return 'Chain (%s)' % ' ->\n'.join([str(n) for n in self.nodes])
