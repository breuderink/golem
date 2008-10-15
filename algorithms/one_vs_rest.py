import logging
import copy
import numpy as np
from dataset import *
import helpers

log = logging.getLogger('OneVsRest')

class OneVsRest:
  def __init__(self, node):
    self.base_node = node

  def train(self, d):
    self.nodes = []
    
    for ci in range(d.nclasses):
      # Create new two-class DataSet
      ys = np.zeros((d.ninstances, 2))
      ys[:, 0] = d.ys[:, ci]
      ys[:, 1] = 1 - d.ys[:, ci]

      curr_d = DataSet(d.xs, ys, d.ids, feature_labels=d.feature_labels, 
        class_labels=['target', 'rest'])

      log.debug(str(curr_d))

      node = copy.deepcopy(self.base_node)
      log.info('Training class %s vs rest' % d.class_labels[ci])
      node.train(curr_d)
      self.nodes.append(node)

  def test(self, d):
    # Evaluate each one_vs_rest classifier
    xs = [n.test(d).xs for n in self.nodes]
    xs = [txs[:,0].reshape(d.ninstances, 1) for txs in xs] # Get target column
    xs = np.hstack(xs)
    return DataSet(xs=xs, ys=d.ys, ids=d.ids, class_labels=d.class_labels)
