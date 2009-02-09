import unittest
import numpy as np
import golem

class TestAUCFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    d = golem.data.gaussian_dataset([200, 200])
    self.d = golem.DataSet(
      xs=np.hstack([d.xs, np.random.random((d.ninstances, 30))]), ys=d.ys)

  def testAUCFilter(self):
    d = self.d
    n = golem.nodes.featsel.AUCFilter()
    n.train(d)
    d2 = n.test(d)
    self.assert_(d2.nfeatures == 2)
    self.assert_(n.feat_bool[:2].all())
  
  def testAUCFilterStrong(self):
    d = self.d
    n = golem.nodes.featsel.AUCFilter(min_auc=.8)
    n.train(d)
    d2 = n.test(d)
    self.assert_(d2.nfeatures == 1)
    self.assert_(n.feat_bool[:1].all())

  def testAUCnfeat(self):
    d = self.d
    for nf in range(d.nfeatures):
      n = golem.nodes.featsel.AUCFilter(nfeatures=nf)
      n.train(d)
      d2 = n.test(d)
      self.assertEqual(d2.nfeatures, nf)
      if nf <= 2:
        self.assert_(n.feat_bool[:nf].all())

  def testAUCFilterTooStrong(self):
    d = self.d
    n = golem.nodes.featsel.AUCFilter(min_auc=1)
    n.train(d)
    d2 = n.test(d)
    self.assert_(d2.nfeatures == 0)
    self.assertFalse(n.feat_bool.all())

  def testAUCFilterSymmetric(self):
    d = self.d
    dn = golem.DataSet(xs=-d.xs, default=d)
    n = golem.nodes.featsel.AUCFilter()
    n.train(d)
    nn = golem.nodes.featsel.AUCFilter()
    nn.train(dn)
    self.assert_((n.feat_bool == nn.feat_bool).all())
