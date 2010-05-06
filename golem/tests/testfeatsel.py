import unittest
import numpy as np
from .. import helpers, plots, DataSet
from ..nodes import featsel

class TestAUCFilter(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    # generate dataset with features based on labels with increasing noise
    ys = helpers.to_one_of_n(np.linspace(0, 1, 1000).round())
    xs = ys[:, 1].reshape(-1, 1) + (np.linspace(.5, 10, 10) * 
      np.random.randn(1000, 10))
    self.d = DataSet(xs=xs, ys=ys, feat_shape=(2, 5))

  def test_AUCFilter(self):
    d = self.d
    n = featsel.AUCFilter()
    self.assertEqual(str(n), 'AUCFilter (using statistic "auc_dev")')

    n.train(d)
    d2 = n.test(d)
    self.assertEqual(d2.nfeatures, 2)
    np.testing.assert_equal(d2.xs, d.xs[:, :2])
    np.testing.assert_equal(n.keep, [0, 1])
    self.assertEqual(str(n), 'AUCFilter (2 features using statistic "auc_dev")')
  
  def test_AUCFilter_strong(self):
    d = self.d
    n = featsel.AUCFilter(min_auc=.8)
    d2 = n.train_apply(d, d)
    self.assertEqual(d2.nfeatures, 1)

  def test_AUCFilter_min_feat(self):
    d = self.d
    for nf in range(d.nfeatures):
      n = featsel.AUCFilter(min_auc=1, min_nfeatures=nf)
      d2 = n.train_apply(d, d)
      self.assertEqual(d2.nfeatures, nf)
      if nf < 4: # higher numbers are too noisy to test reliably
        self.assertEqual(n.keep.tolist(), range(nf))

  def test_AUCFilter_auc_nfeat_combo(self):
    # test with more strict min_auc
    d = self.d
    n = featsel.AUCFilter(min_auc=.6, min_nfeatures=1)
    d2 = n.train_apply(d, d)
    self.assertEqual(d2.nfeatures, 2)

    # test with more strict min_nfeatures
    n = featsel.AUCFilter(min_auc=.6, min_nfeatures=6)
    d2 = n.train_apply(d, d)
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
        np.testing.assert_equal(n.keep, nn.keep)
