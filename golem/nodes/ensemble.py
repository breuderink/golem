import logging, copy, itertools
import numpy as np
from ..dataset import DataSet
from .. import helpers

def copy_splitter(d):
  while True:
    yield d

def bagging_splitter(d):
  while True:
    yield d[np.random.random_integers(0, d.ninstances, d.ninstances)]

def average_combiner(ds):
  xs = np.zeros(ds[0].xs.shape)
  for d in ds:
    xs += d.xs
  return DataSet(xs=xs/float(len(ds)), default=ds[0])

class Ensemble:
  def __init__(self, nodes, tr_splitter=copy_splitter, 
    te_splitter=copy_splitter, combiner=average_combiner):
    assert isinstance(nodes, list)
    self.nodes = nodes 
    self.tr_splitter = tr_splitter
    self.te_splitter = te_splitter
    self.combiner = combiner
  
  def train(self, d):
    for (n, nd) in itertools.izip(self.nodes, self.tr_splitter(d)):
      n.train(nd)

  def test(self, d):
    xs = np.zeros(d.xs.shape)
    results = [n.test(nd) for (n, nd) in itertools.izip(self.nodes, 
      self.te_splitter(d))]
    assert len(results) == len(self.nodes), 'Not all nodes were used'
    return self.combiner(results)

class OVONode:
  def __init__(self, cia, cib, node):
    self.cia = cia
    self.cib = cib
    self.node = node

  def train(self, d):
    cia, cib = self.cia, self.cib
    pair_d = d.get_class(self.cia) + d.get_class(self.cib)
    ys = pair_d.ys[:, [cia, cib]]
    cl_lab = [d.cl_lab[cia], d.cl_lab[cib]]
    self.node.train(DataSet(ys=ys, cl_lab=cl_lab, default=pair_d))

  def test(self, d):
    cia, cib = self.cia, self.cib
    td = self.node.test(DataSet(ys=d.ys[:,[cia, cib]], cl_lab=[d.cl_lab[cia], 
      d.cl_lab[cib]], default=d))
    xs = np.zeros((d.ninstances, d.nclasses))
    xs[:, [self.cia, self.cib]] = td.xs
    return DataSet(xs=xs, default=d)

class OneVsOne:
  def __init__(self, base_node):
    self.base_node = base_node
  
  def train(self, d):
    pairs = []
    for cia in range(d.nclasses):
      for cib in range(cia + 1, d.nclasses):
        pairs.append(OVONode(cia, cib, copy.deepcopy(self.base_node)))

    self.ensemble = Ensemble(pairs)
    self.ensemble.train(d)

  def test(self, d):
    return self.ensemble.test(d)

class OVRNode:
  def __init__(self, class_i, node):
    self.class_i = class_i
    self.node = node

  def ovr_d(self, d):
    class_i = self.class_i
    ys = np.zeros((d.ninstances, 2))
    ys[:, 0] = d.ys[:, class_i]
    ys[:, 1] = 1 - d.ys[:, class_i]
    return DataSet(ys=ys, cl_lab=[d.cl_lab[class_i], 'rest'], default=d)

  def train(self, d):
    self.node.train(self.ovr_d(d))

  def test(self, d):
    td = self.node.test(self.ovr_d(d))
    xs = []
    for i in range(d.nclasses):
      if i == self.class_i:
        xs.append(td.xs[:, 0].reshape(-1, 1))
      else:
        xs.append(td.xs[:, 1].reshape(-1, 1))
    return DataSet(xs=np.hstack(xs), default=d)

class OneVsRest:
  def __init__(self, base_node):
    self.base_node = base_node
  
  def train(self, d):
    nodes = [OVRNode(ci, copy.deepcopy(self.base_node)) for ci in 
      range(d.nclasses)]
    self.ensemble = Ensemble(nodes)
    self.ensemble.train(d)

  def test(self, d):
    return self.ensemble.test(d)

class Bagging(Ensemble):
  def __init__(self, n, base_node):
    Ensemble.__init__(self, [copy.deepcopy(base_node) for i in range(n)])
