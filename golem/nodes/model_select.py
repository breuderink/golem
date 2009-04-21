import copy
import logging

log = logging.getLogger('golem.ModelSelect')

class ModelSelect:
  def __init__(self, nodes, critic):
    assert(isinstance(nodes, list))
    self.nodes = nodes
    self.critic = critic
    self.best_node = None

  def train(self, d):
    best_node = None
    for node in self.nodes:
      log.info('Evaluating: %s' % str(node))
      perf = self.critic(d, copy.deepcopy(node))
      if best_node == None or perf > best_perf:
        best_perf = perf
        best_node = node
    best_node.train(d) 
    self.best_node = best_node

  def test(self, d):
    return self.best_node.test(d)

  def __str__(self):
    if self.best_node <> None:
      return 'ModelSelect (best_node = %s)' % self.best_node
    return 'ModelSelect (untrained)'
