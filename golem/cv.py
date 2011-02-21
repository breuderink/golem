import copy
import logging
import numpy as np
from dataset import DataSet
import helpers

log = logging.getLogger('golem.cv')

def cross_validate(subsets, node):
  """
  Cross-validate on subsets using node. Returns a list with the output of node
  on the test sets. It can give an estimate of how well the analysis will work
  in practice [1].

  This methods has a few safety measures:
  - for every fold, we start with a fresh copy of node
  - the labels of the test sets are removed to prevent node.test() from cheating
  
  Example of use:
    
  >>>  cross_validate(seq_splits(d, 5), n)
    
  This applies node n to the train and test sets based on the 
  sequential split of dataset d into 5 subsets, and returns the output for each.  
  Cross validation is used to see how a certain analysis generalizes from one
  subset of the data to another, without needing an explicit validation dataset. 

  [1] http://en.wikipedia.org/wiki/Cross-validation_(statistics)
  """
  for (tr, te) in cross_validation_sets(subsets):
    # fresh copy, no cheating by remembering
    inode = copy.deepcopy(node)

    inode.train(tr)

    # create a test set stripped of labels
    te_stripped = DataSet(ys=np.nan * te.ys, default=te)

    pred = inode.apply(te_stripped)

    # reattach labels
    pred = DataSet(ys=te.ys, default=pred)

    del inode, tr, te
    yield pred


def rep_cv(d, node, reps=5, K=10):
  """
  Repeated cross-validation shuffled stratified subsets of d. 
  Returns a list with the output of node on the test sets.
  """
  for ri in range(reps):
    for td in cross_validate(strat_splits(d, K), node):
      yield td
def strat_splits(d, K=10):
  """
  Splits a dataset in K non-overlapping subsets. The classes are distributed
  evenly over the subsets.
  """
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
  return [s.sorted() for s in subsets] # Remove class-relevant order

def seq_splits(d, K=10):
  """
  Splits a dataset in K non-overlapping subsets. The first subset is created
  from the first Kth part of d, the second subset from the second Kth part of
  d etc.
  
  For data that is time-dependent, the cross-validation results on these data
  sets will be more representative than with strat_splits().
  """
  assert(K <= d.ninstances)
  indices = np.floor(np.linspace(0, K, d.ninstances, endpoint=False))
  result = []
  for i in range(K):
    result.append(d[indices==i])
  return result

def cross_validation_sets(subsets):
  """
  Generate training and test sets from a list with DataSets. The training set
  is created from the subsets after isolating a test set.
  
  Returns a list of training sets and a list of test sets. The first test
  set is the first subset. The first training set are all the subsets except
  for the first subset. The second test set is the second subset, etc.
  
  To apply cross-validation, it is recommended to use cross_validate().
  """
  K = len(subsets)
  for ki in range(K):
    training_set = (reduce(lambda a, b: a + b, 
      [subsets[i] for i in range(len(subsets)) if i <> ki]))
    test_set = subsets[ki]
    log.info('Building training and testset %d of %d' % (ki + 1, K))
    yield training_set, test_set

