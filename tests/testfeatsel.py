import unittest
import numpy as np
import golem

class TestAUCFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    d = golem.data.gaussian_dataset([200, 200])
    self.d = golem.DataSet(
      xs=np.hstack([d.xs, np.random.random((d.ninstances, 30))]), ys=d.ys)
    print self.d

  def testAUCFilter(self):
    d = self.d
    n = golem.nodes.featsel.AUCFilter()
    n.train(d)
    d2 = n.test(d)
    self.assert_(d2.nfeatures == 2)
    self.assert_(n.feat_bool[:2].all())
