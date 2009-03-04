import logging
import copy
import numpy as np
from ..dataset import DataSet
from .. import helpers

log = logging.getLogger('OneVsRest')

class OneVsRest:
  def __init__(self, node):
    self.base_node = node
    self.nodes = None

  def train(self, d):
    self.nodes = []
    
    for ci in range(d.nclasses):
      # Create new two-class DataSet
      ys = np.zeros((d.ninstances, 2))
      ys[:, 0] = d.ys[:, ci]
      ys[:, 1] = 1 - d.ys[:, ci]

      curr_d = DataSet(ys=ys, cl_lab=[d.cl_lab[ci], 'rest'], default=d)

      log.debug(str(curr_d))

      node = copy.deepcopy(self.base_node)
      log.info('Training class %s vs rest' % d.cl_lab[ci])
      node.train(curr_d)
      self.nodes.append(node)
      self.cl_lab = d.cl_lab

  def test(self, d):
    # Evaluate each one_vs_rest classifier
    xs = [n.test(d).xs for n in self.nodes]
    xs = [txs[:,0].reshape(d.ninstances, 1) for txs in xs] # Get target column
    xs = np.hstack(xs)
    return DataSet(xs, feat_lab=self.cl_lab, default=d)

  def __str__(self):
    if self.nodes == None:
      return 'OneVsRest (%s)' % self.base_node
    else:
      return 'OneVsRest (%s)' % ', '.join([str(n) for n in self.nodes])
