import unittest
import numpy as np
from .. import stats

class TestHolmsProcedure(unittest.TestCase):
  def setUp(self):
    np.random.seed(2)

  def test_fp(self):
    '''
    Test for conservativeness
    '''
    for alpha in [0.01, .05]:
      for n in [3, 10, 100]:
        fps = [np.any(stats.holms_proc(np.random.rand(10), alpha))
          for i in range(1000)]
        self.assert_(np.mean(fps) / alpha <= 1,
          'FPR=%.5f, alpha=%.2f' % (np.mean(fps), alpha))

  def test_sensitivity(self):
    print
    for alpha in [0.01, .05]:
      for n in [3, 10, 100]:
        for i in range(10000):
          tps = []
          p = np.random.rand(n)
          p[2] = np.random.rand() * alpha
          tps.append(stats.holms_proc(p)[2])
        print 'alpha:%.2f, n:%.2f, TPR:%.4f' % (alpha, n, np.mean(tps))
    
