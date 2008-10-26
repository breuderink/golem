import unittest
import copy
import logging
import numpy as np

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
  #logging.getLogger('OneVsOne').setLevel(logging.INFO)
  d = helpers.artificialdata.gaussian_dataset([60, 60, 60])  
  n = nodes.ModelSelect(gen_svms(), cv_acc_critic)
 
  results = [r for r in cross_validate(stratified_split(d, 5), n)]
  accs = [loss.accuracy(r) for r in results]
  print np.mean(accs)
  print loss.format_confmat(reduce(lambda a, b: a + b, results))

  n.train(d) 
  print n.test(d)
  print n
  helpers.plots.scatter_plot(d)
  helpers.plots.plot_classifier_hyperplane(n.best_node.nodes[(0, 1)], '01.png')
  helpers.plots.scatter_plot(d)
  helpers.plots.plot_classifier_hyperplane(n.best_node.nodes[(0, 2)], '02.png')
  helpers.plots.scatter_plot(d)
  helpers.plots.plot_classifier_hyperplane(n.best_node.nodes[(1, 2)], '12.png')
  
  helpers.plots.scatter_plot(d)
  helpers.plots.plot_classifier_hyperplane(n, 'mclass.png')
