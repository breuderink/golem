import logging, copy, itertools
import numpy as np
from ..dataset import DataSet
from .. import helpers
from basenode import BaseNode

# returns ref to the dataset
def copy_splitter(d):
  while True:
    yield d

# returns new dataset with average of the xs in ds
def average_combiner(ds):
  xs = np.zeros(ds[0].xs.shape)
  for d in ds:
    xs += d.xs
  return DataSet(xs=xs/float(len(ds)), default=ds[0])

# separate splitter for training and test data
class Ensemble(BaseNode):
  def __init__(self, nodes, tr_splitter=copy_splitter, 
    te_splitter=copy_splitter, combiner=average_combiner):
    BaseNode.__init__(self)
    assert isinstance(nodes, list)
    self.nodes = nodes 
    self.tr_splitter = tr_splitter
    self.te_splitter = te_splitter
    self.combiner = combiner
  
  def train_(self, d):
    for (n, nd) in itertools.izip(self.nodes, self.tr_splitter(d)):
      n.train(nd)

  def apply_(self, d):
    # xs = np.zeros(d.xs.shape)
    results = [n.apply(nd) for (n, nd) in itertools.izip(self.nodes, 
      self.te_splitter(d))]
    return self.combiner(results)

class OVONode(BaseNode):
  def __init__(self, cia, cib, node):
    BaseNode.__init__(self)
    self.cia = cia
    self.cib = cib
    self.node = node

  def train_(self, d):
    cia, cib = self.cia, self.cib
    pair_d = d.get_class(self.cia) + d.get_class(self.cib)
    ys = pair_d.ys[:, [cia, cib]]
    cl_lab = [d.cl_lab[cia], d.cl_lab[cib]]
    self.node.train(DataSet(ys=ys, cl_lab=cl_lab, default=pair_d))

  def apply_(self, d):
    cia, cib = self.cia, self.cib
    td = self.node.apply(DataSet(ys=d.ys[:,[cia, cib]], cl_lab=[d.cl_lab[cia], 
      d.cl_lab[cib]], default=d))
    xs = np.zeros((d.ninstances, d.nclasses))
    xs[:, [self.cia, self.cib]] = td.xs
    return DataSet(xs=xs, default=d)

# binary node can distinguish between 2 classes
class OneVsOne(BaseNode):
  def __init__(self, binary_node):
    BaseNode.__init__(self)
    self.binary_node = binary_node
  
  # for each pair, train classifier
  def train_(self, d):
    pairs = []
    for cia in range(d.nclasses):
      for cib in range(cia + 1, d.nclasses):
        pairs.append(OVONode(cia, cib, copy.deepcopy(self.binary_node)))

    self.ensemble = Ensemble(pairs)
    self.ensemble.train(d)

  def apply_(self, d):
    return self.ensemble.apply(d)

class OVRNode(BaseNode):
  def __init__(self, class_i, node):
    BaseNode.__init__(self)
    self.class_i = class_i
    self.node = node

  def ovr_d(self, d):
    class_i = self.class_i
    ys = np.zeros((d.ninstances, 2))
    ys[:, 0] = d.ys[:, class_i]
    ys[:, 1] = 1 - d.ys[:, class_i]
    return DataSet(ys=ys, cl_lab=[d.cl_lab[class_i], 'rest'], default=d)

  def train_(self, d):
    self.node.train(self.ovr_d(d))

  def apply_(self, d):
    td = self.node.apply(self.ovr_d(d))
    xs = []
    for i in range(d.nclasses):
      if i == self.class_i:
        xs.append(td.xs[:, 0].reshape(-1, 1))
      else:
        xs.append(td.xs[:, 1].reshape(-1, 1))
    return DataSet(xs=np.hstack(xs), default=d)

class OneVsRest(BaseNode):
  def __init__(self, binary_node):
    BaseNode.__init__(self)
    self.binary_node = binary_node
  
  def train_(self, d):
    nodes = [OVRNode(ci, copy.deepcopy(self.binary_node)) for ci in 
      range(d.nclasses)]
    self.ensemble = Ensemble(nodes)
    self.ensemble.train(d)

  def apply_(self, d):
    return self.ensemble.apply(d)

# return random subsets of your dataset
def bagging_splitter(d):
  while True:
    i = np.random.random_integers(0, d.ninstances-1, d.ninstances)
    yield DataSet(xs=d.xs[i, :], ys=d.ys[i,:], ids=None, default=d)

class Bagging(Ensemble):
  def __init__(self, base_node, n):
    Ensemble.__init__(self, [copy.deepcopy(base_node) for i in range(n)], 
      tr_splitter=bagging_splitter)
