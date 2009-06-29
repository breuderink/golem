import operator
import numpy as np
from ..cv import stratified_split

class Chain:
  def __init__(self, nodes):
    assert isinstance(nodes, list)
    self.nodes = nodes
      
  def train(self, d):
    for (i, n) in enumerate(self.nodes[:-1]):
      n.train(d)
      d = n.test(d)
    self.nodes[-1].train(d)
    
  def test(self, d):
    for n in self.nodes:
      d = n.test(d)
    return d

  def __str__(self):
    return 'Chain (%s)' % ' -> '.join([str(n) for n in self.nodes])

class RationedChain:
  def __init__(self, parts, nodes):
    assert isinstance(nodes, list)
    assert isinstance(parts, list)
    assert len(parts) == len(nodes)
    self.nodes = nodes
    self.parts = parts
    self.pis = [range(s, e) for (s, e) in
      zip(np.cumsum([0] + parts[:-1]), np.cumsum(parts))]
      
  def train(self, d):
    ds = stratified_split(d, K=np.sum(self.parts))
    ds = [reduce(operator.add, [ds[i] for i in pi]) for pi in self.pis]
    
    for (i, tr_n) in enumerate(self.nodes):
      cd = ds[i]
      for te_n in self.nodes[:i]:
        cd = te_n.test(cd)
      tr_n.train(cd)
    
  def test(self, d):
    for n in self.nodes:
      d = n.test(d)
    return d

  def __str__(self):
    return 'RationedChain (%s)' % ' -> '.join([str(n) for n in self.nodes])
