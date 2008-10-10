import unittest

import numpy as np
from dataset import *
from helpers import *
from pprint import pprint

def stratified_split(d, K=10):
  subsets = []
  for ci in range(d.nclasses):
    cid = d.get_class(ci)
    ind = np.arange(cid.ninstances) % K
    for si in range(K):
      indices = np.where(ind == si)[0].copy().tolist()
      if si < len(subsets):
        subsets[si] += cid[indices]
      else:
        subsets.append(cid[indices])
  return subsets

def cross_validation_sets(subsets):
  K = len(subsets)
  for ki in range(K):
    training_set = (reduce(lambda a, b: a + b, 
      [subsets[i] for i in range(len(subsets)) if i <> ki]))
    test_set = subsets[ki]
    yield training_set, test_set

def cross_validate(subsets, node):
  for (tr, te) in cross_validation_sets(subsets):
    node.train(tr)
    yield node.test(te)

if __name__ == '__main__':
  #unittest.main()
  import algorithms
  import loss

  np.random.seed(1)
  d = artificialdata.gaussian_dataset([300, 200])
  K = 5
  
  svm = algorithms.SupportVectorMachine(C=10)
  test_folds = [d for d in cross_validate(stratified_split(d, K), svm)]
  accuracy = [loss.accuracy(d) for d in test_folds]
  print np.mean(accuracy), np.std(accuracy)

