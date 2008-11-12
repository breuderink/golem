import logging
import copy
import numpy as np
from golem import DataSet, helpers
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
      
        # Create new two-class DataSet
        pair_d = d.get_class(cia) + d.get_class(cib)
        ys = pair_d.ys[:, [cia, cib]]
        cl_lab = [d.cl_lab[cia], d.cl_lab[cib]]
        log.info('Training class %s vs class %s' % (cl_lab[0], cl_lab[1]))
        pair_d = DataSet(ys=ys, cl_lab=cl_lab, default=pair_d)
        node = copy.deepcopy(self.base_node)
        node.train(pair_d)
        self.nodes[(cia, cib)] = node

    log.info('Done.')
    self.cl_lab = d.cl_lab

  def test(self, d):
    xs = np.zeros((d.ninstances, self.nclasses))
    for (cia, cib) in self.nodes:
      pred = self.nodes[(cia, cib)].test(d)
      xs[:, cia] += pred.xs[:, 0]
      xs[:, cib] += pred.xs[:, 1]
    return DataSet(xs, feat_lab=self.cl_lab, default=d)

  def __str__(self):
    return 'OneVsOne (%s)' % self.base_node
