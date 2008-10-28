import unittest
from dataset import *
from helpers import artificialdata
import nodes
import loss

class TestLSReg(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    self.d = artificialdata.gaussian_dataset([200, 200, 200])

  def test_perf(self):
    '''Test LSReg'''
    n = nodes.LSReg()
    n.train(self.d)
    self.assert_(loss.accuracy(n.test(self.d)) > 0.8)

  def test_1feature(self):
    '''Test LSReg with 1 feature'''
    d = self.d
    d = DataSet(d.xs[:, 0].reshape(d.ninstances, 1), d.ys, d.ids)
    n = nodes.LSReg()
    n.train(d)
    self.assert_(loss.accuracy(n.test(d)) > 0.6)
    
