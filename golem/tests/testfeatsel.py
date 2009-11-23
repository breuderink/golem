import unittest
import numpy as np
from .. import helpers, plots, DataSet
from ..nodes import featsel

class TestAUCFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    # generate dataset with features based on labels with increasing noise
    ys = helpers.to_one_of_n(np.linspace(0, 1, 400).round())
    xs = ys[:, 1].reshape(-1, 1) + (np.linspace(.5, 10, 10) * 
      np.random.randn(400, 10))
    self.d = DataSet(xs=xs, ys=ys, feat_shape=(2, 5))

  def test_AUCFilter(self):
    d = self.d
    n = featsel.AUCFilter()
    self.assertEqual(str(n), 'AUCFilter (using statatistic "onesided_auc")')

    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 2)
    np.testing.assert_equal(d2.xs, d.xs[:, :2])
    np.testing.assert_equal(n.feat_bool, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(str(n), 
      'AUCFilter (2 of 10 features using statistic "onesided_auc")')
  
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
    # test with more strict min_auc
    d = self.d
    n = featsel.AUCFilter(min_auc=.6, min_nfeatures=1)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 2)

    # test with more strict min_nfeatures
    n = featsel.AUCFilter(min_auc=.6, min_nfeatures=6)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 6)

  def test_AUCFilter_too_strong(self):
    d = self.d
    n = featsel.AUCFilter(min_auc=1)
    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 0)

  def test_AUCFilter_is_symmetric(self):
    d = self.d
    dn = DataSet(xs=-d.xs, default=d)
    for min_auc in [.5, .6, .9, 1]:
      for nf in range(10):
        n = featsel.AUCFilter(min_auc=min_auc, min_nfeatures=nf)
        nn = featsel.AUCFilter(min_auc=min_auc, min_nfeatures=nf)
        n.train(d)
        nn.train(dn)
        np.testing.assert_equal(n.feat_bool, nn.feat_bool)
