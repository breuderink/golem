import copy
import logging
import numpy as np
from dataset import DataSet
import helpers

log = logging.getLogger('CV')
def strat_splits(d, K=10):
  '''
  Splits a dataset in K non-overlapping subsets. The classes are distributed
  evenly over the subsets.
  '''
  subsets = []
  assert(K <= max(d.ninstances_per_class))
  d = d.shuffled()
  # Loop over classes
  for ci in range(d.nclasses):
    cid = d.get_class(ci)
    ind = np.arange(cid.ninstances) % K
    for si in range(K):
      # Loop over future subsets
      if si < len(subsets):
        subsets[si] += cid[ind == si] # Grow subset
      else:
        subsets.append(cid[ind == si]) # Create subset
  return subsets

def seq_splits(d, K=10):
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
  This methods has a few safety measures:
  - for every fold, we start with a fresh copy of node
  - the test method of the trained node can *never* see the labels
  '''
  for (tr, te) in cross_validation_sets(subsets):
    # fresh copy, no cheating by remembering
    tnode = copy.deepcopy(node)

    tnode.train(tr)

    # create a test set stripped of labels
    te_stripped = DataSet(ys=np.zeros(te.ys.shape), default=te)

    pred = tnode.test(te_stripped)

    # reattach labels
    pred = DataSet(ys=te.ys, default=pred)

    del tnode
    yield pred


def rep_cv(d, node, reps=5, K=10):
  '''
  Repeated cross-validation shuffled stratified subsets of d. 
  Returns a list with the output of node on the testsets.
  '''
  for ri in range(reps):
    for td in cross_validate(strat_splits(d), node):
      yield td
