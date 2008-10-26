import logging
import copy
import numpy as np
from dataset import *
import helpers

log = logging.getLogger('OneVsOne')

class OneVsOne:
  def __init__(self, node):
    self.base_node = node
    self.nodes = None

  def train(self, d):
    self.nodes = {}
    
    self.nclasses = d.nclasses
    for cia in range(d.nclasses):
      for cib in range(cia + 1, d.nclasses):
        log.info('Training class %s vs class %s' % 
          (d.class_labels[cia], d.class_labels[cib]))
      
        # Create new two-class DataSet
        pair_d = d.get_class(cia) + d.get_class(cib)
        ys = pair_d.ys[:, [cia, cib]]
        cl = [pair_d.class_labels[i] for i in range(len(pair_d.class_labels))
          if i in [cia, cib]]
        pair_d = DataSet(pair_d.xs, ys, pair_d.ids,
          feature_labels=pair_d.feature_labels, class_labels=cl)
        node = copy.deepcopy(self.base_node)
        node.train(pair_d)
        self.nodes[(cia, cib)] = node

    log.info('Done.')

  def test(self, d):
    xs = np.zeros((d.ninstances, self.nclasses))
    for (cia, cib) in self.nodes:
      pred = self.nodes[(cia, cib)].test(d)
      xs[:, cia] += pred.xs[:, 0]
      xs[:, cib] += pred.xs[:, 1]
    return DataSet(xs, d.ys, d.ids, class_labels=d.class_labels)

  def __str__(self):
    return 'OneVsOne (%s)' % self.base_node
