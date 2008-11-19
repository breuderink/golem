import unittest
import copy
import logging
import numpy as np
import pylab 

import golem as g

def gen_svms():
  result = []
  for sigma in [.1, 1, 10]:
    for C in [1, 10, 100, 1000]:
      result.append(g.nodes.OneVsRest(
        g.nodes.SVM(C=C, kernel='rbf', sigma=sigma)))
  return result
      
def cv_acc_critic(d, node):
  splits = g.crossval.stratified_split(d, 5)
  return np.mean([g.loss.accuracy(r) for r in g.crossval.cross_validate(splits, node)])
  
if __name__ == '__main__':
  np.random.seed(1)
  logging.basicConfig(level=logging.INFO)

  d = g.data.gaussian_dataset([400, 400, 300])
  n = g.nodes.ModelSelect(gen_svms(), cv_acc_critic)

  aucs = [g.loss.accuracy(dt) for dt in \
    g.crossval.cross_validate(g.crossval.stratified_split(d, 4), n)]
  print aucs
