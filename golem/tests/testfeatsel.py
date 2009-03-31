import unittest
import numpy as np
from .. import data, helpers, plots, DataSet
from ..nodes import featsel

class TestAUCFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    d = data.gaussian_dataset([200, 200])
    self.d = DataSet(
      xs=np.hstack([d.xs, np.random.random((d.ninstances, 30))]), ys=d.ys)

  def test_AUCFilter(self):
    d = self.d
    n = featsel.AUCFilter()
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 2)
    self.assert_(n.feat_bool[:2].all())
  
  def test_AUCFilter_strong(self):
    d = self.d
    n = featsel.AUCFilter(min_auc=.8)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 1)

  def test_AUCFilter_min_feat(self):
    d = self.d
    for nf in range(d.nfeatures):
      n = featsel.AUCFilter(min_auc=1, min_nfeatures=nf)
      n.train(d)
      d2 = n.test(d)
      self.assertEqual(d2.nfeatures, nf)
      if nf <= 2:
        self.assert_(n.feat_bool[:nf].all())

  def test_AUCFilter_auc_nfeat_combo(self):
    d = self.d
    n = featsel.AUCFilter(min_auc=.6, min_nfeatures=1)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 2)

  def test_AUCFilterr_too_strong(self):
    d = self.d
    n = featsel.AUCFilter(min_auc=1)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 0)
    self.assertFalse(n.feat_bool.all())

  def test_AUCFilter_is_symmetric(self):
    d = self.d
    dn = DataSet(xs=-d.xs, default=d)
    n = featsel.AUCFilter(min_nfeatures=3)
    n.train(d)
    nn = featsel.AUCFilter(min_nfeatures=3)
    nn.train(dn)
    np.testing.assert_equal(n.feat_bool, nn.feat_bool)
