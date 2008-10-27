import unittest
import copy
import logging
import numpy as np
import pylab 

from helpers import *
from crossval import *
import nodes
import loss
    
def gen_svms():
  result = []
  for sigma in [.1, .3, .5, .8, 1, 2, 5]:
    for C in [1, 10, 50, 100, 200, 2000]:
      result.append(nodes.OneVsOne(nodes.SVM(C=C, kernel='rbf', sigma=sigma)))
      #result.append(nodes.OneVsRest(nodes.SVM(C=C, kernel='linear')))
  return result
      
def cv_acc_critic(d, node):
  splits = stratified_split(d, 5)
  return np.mean([loss.accuracy(r) for r in cross_validate(splits, node)])
  
if __name__ == '__main__':
  np.random.seed(1)
  logging.basicConfig(level=logging.WARNING)
  logging.getLogger('PCA').setLevel(logging.DEBUG)
 
  xs = np.random.rand(100, 3)
  xs = np.hstack([xs, -xs, np.zeros((100, 2))]) # make correlated
  d = DataSet(xs, np.ones((100, 1)), None)

  #n = nodes.PCA(retain=.99)
  n = nodes.PCA(ndims=4)
  print n
  n.train(d)
  print n
  d2 = n.test(d)

  cov = np.cov(d2.xs, rowvar=False)
  cov_target=np.diag(np.diag(cov)) # construct a diagonal uncorrelated cov
  assert(((cov - cov_target) ** 2 < 1e-8).all())

  pylab.matshow(cov)
  pylab.show()
