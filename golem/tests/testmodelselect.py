import unittest
import copy
import logging
import numpy as np

from .. import nodes, data

def svm_critic(d, node):
  return 1 if isinstance(node, nodes.SVM) else 0

class TestModelSelection(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    self.d = data.gaussian_dataset([60, 60])  

  def test_select_best(self):
    n = nodes.ModelSelect([nodes.PriorClassifier(), nodes.SVM()], svm_critic)
    n.train(self.d) 
    s = nodes.SVM()
    s.train(self.d)
    self.assert_(isinstance(n.best_node, nodes.SVM))
    self.assertEqual(n.test(self.d), s.test(self.d))
