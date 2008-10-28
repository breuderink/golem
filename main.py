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
  logging.getLogger('LSReg').setLevel(logging.DEBUG)

  unittest.main()
  d = artificialdata.gaussian_dataset([400, 400, 300])

  #plots.scatter_plot(d)
  #plots.plot_classifier_hyperplane(n, fname='lsreg.png')
