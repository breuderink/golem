import copy
import logging

from basenode import BaseNode

class ModelSelect(BaseNode):
  def __init__(self, nodes, critic):
    BaseNode.__init__(self)
    self.nodes = list(nodes)
    self.critic = critic

  def train_(self, d):
    best_node = None
    for node in self.nodes:
      self.log.info('Evaluating: %s' % str(node))
      perf = self.critic(d, copy.deepcopy(node))
      if best_node == None or perf > best_perf:
        best_perf = perf
        best_node = node
    best_node.train(d) 
    self.best_node = best_node
    self.log.info('Best node: %s' % str(best_node))

  def test_(self, d):
    return self.best_node.test(d)

  def __str__(self):
    if hasattr(self, 'best_node'):
      return 'ModelSelect (selected %s)' % self.best_node
    return 'ModelSelect'
