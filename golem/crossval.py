import copy
import logging
import numpy as np
from golem import DataSet, helpers

log = logging.getLogger('CV')
def stratified_split(d, K=10):
  '''
  Splits a dataset in K non-overlapping subsets. The classes are distributed
  evenly over the subsets.
  '''
  subsets = []
  assert(K <= max(d.ninstances_per_class))
  # Loop over classes
  for ci in range(d.nclasses):
    cid = d.get_class(ci)
    ind = np.arange(cid.ninstances) % K
    for si in range(K):
      # Loop over future subsets
      indices = np.where(ind == si)[0].copy().tolist()
      if si < len(subsets):
        subsets[si] += cid[indices] # Grow subset
      else:
        subsets.append(cid[indices]) # Create subset
  return subsets

def sequential_split(d, K=10):
  '''
  Splits a dataset in K non-overlapping subsets. The first subset is created
  from the first Kth part of d, the second subset from thet second Kth part of
  d etc.
  '''
  assert(K <= d.ninstances)
  indices = np.floor(np.linspace(0, K, d.ninstances, endpoint=False))
  result = []
  for i in range(K):
    result.append(d[indices==i])
  return result

def cross_validation_sets(subsets):
  '''
  Generete training and testsets from a list with DataSets. The trainingset
  is created from the subsets after isolating a testset.
  '''
  K = len(subsets)
  for ki in range(K):
    training_set = (reduce(lambda a, b: a + b, 
      [subsets[i] for i in range(len(subsets)) if i <> ki]))
    test_set = subsets[ki]
    log.info('Building training and testset %d of %d' % (ki, K))
    yield training_set, test_set

def cross_validate(subsets, node):
  '''
  Crossvalidate on subsets using node. Returns a list with the output of node
  on the testsets.
  '''
  for (tr, te) in cross_validation_sets(subsets):
    tnode = copy.deepcopy(node) # To be sure we don't cheat
    tnode.train(tr)
    yield tnode.test(te)
    del tnode
