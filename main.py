import unittest
import copy
import numpy as np
from helpers import *
import algorithms as alg
from crossval import *
import loss
import logging

log = logging.getLogger('SelectBest')

def cv_acc_critic(d, node):
  splits = stratified_split(d, 5)
  return np.mean([loss.accuracy(r) for r in cross_validate(splits, node)])

class SelectBest:
  def __init__(self, nodes, critic):
    assert(isinstance(nodes, list))
    self.nodes = nodes
    self.critic = critic
    self.best_node = None

  def train(self, d):
    best_perf = 0
    for node in self.nodes:
      log.info('Evaluating: %s' % str(node))
      perf =  self.critic(d, copy.deepcopy(node))
      if perf > best_perf:
        best_perf = perf
        best_node = node
    best_node.train(d) 
    self.best_node = best_node

  def test(self, d):
    return self.best_node.test(d)

  def __str__(self):
    if self.best_node <> None:
      return 'SelectBest (best_node = %s)' % self.best_node
    return 'SelectBest (untrained)'

def gen_svms():
  result = []
  #for sigma in [.1, .3, .5, .8, 1, 2, 5]:
  for C in [1, 10, 50, 100, 200, 2000]:
    #result.append(alg.SVM(C=C, kernel='rbf', sigma=sigma))
    result.append(alg.SVM(C=C, kernel='linear'))
  return result
      
  
if __name__ == '__main__':
  np.random.seed(1)
  logging.basicConfig(level=logging.WARNING)
  d = helpers.artificialdata.gaussian_dataset([60, 60, 60])  
  
  n = alg.Chain([alg.ZScore(), 
    alg.OneVsRest(SelectBest(gen_svms(), cv_acc_critic))])
  print n
  results = [r for r in cross_validate(stratified_split(d, 5), n)]
  accs = [loss.accuracy(r) for r in results]
  
  print np.mean(accs)
  print loss.format_confmat(reduce(lambda a, b: a + b, results))

  n.train(d) 
  print n
  helpers.plots.scatter_plot(d)
  helpers.plots.plot_classifier_hyperplane(n, 'mclass.png')
