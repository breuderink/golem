import unittest
import numpy as np
from .. import DataSet, perf, helpers

class TestPerf(unittest.TestCase):
  def setUp(self):
    np.random.seed(1)
    self.d = DataSet(
      xs=helpers.to_one_of_n([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]),
      ys=helpers.to_one_of_n([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))

  def test_class_loss(self):
    targets = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1)
    d = self.d
    d2 = DataSet(xs=d.xs + (0.1 * np.random.rand(*d.xs.shape)), default=d)
    np.testing.assert_equal(perf.class_loss(d), targets)
    np.testing.assert_equal(perf.class_loss(d2), targets)
  
  def test_accurracy(self):
    self.assertEqual(perf.accuracy(self.d), (12 - 3) / 12.)

  def test_conf_mat(self):
    c = perf.conf_mat(self.d)
    ct = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]])
    np.testing.assert_equal(c, ct)
  
  def test_format_confusion_matrix(self):
    c = perf.format_confmat(perf.conf_mat(self.d), self.d)
    target = [['Label\\Pred.', 'class0', 'class1', 'class2'],
      ['class0', 3, 1, 0],
      ['class1', 0, 3, 1],
      ['class2', 1, 0, 3]]
    self.assertEqual(c, target)
  
  def test_AUC(self):
    d1 = DataSet(helpers.to_one_of_n([0, 0, 1, 1, 1, 1]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    d0 = DataSet(helpers.to_one_of_n([1, 1, 0, 0, 0, 0]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    dr = DataSet(helpers.to_one_of_n([1, 0, 1, 0, 1, 0]),
      helpers.to_one_of_n([1, 1, 1, 1, 0, 0]))

    self.assertEqual(perf.auc(d0), 0)
    self.assertEqual(perf.auc(d1), 1)
    self.assertEqual(perf.auc(dr), .5)

  def test_I(self):
    d1 = DataSet(helpers.to_one_of_n([0, 0, 0, 1, 1, 1]),
      helpers.to_one_of_n([0, 0, 0, 1, 1, 1]))
    d0 = DataSet(helpers.to_one_of_n([1, 1, 1, 0, 0, 0]),
      helpers.to_one_of_n([0, 0, 0, 1, 1, 1]))
    dr = DataSet(helpers.to_one_of_n([1, 0, 1, 0, 1, 0]),
      helpers.to_one_of_n([1, 1, 1, 1, 0, 0]))

    self.assertAlmostEqual(perf.I(d1), 1)
    self.assertAlmostEqual(perf.I(d0), 1)
    self.assertAlmostEqual(perf.I(dr), 0)

  def test_mean_std(self):
    ds = [DataSet(xs=np.random.rand(30, 2), ys=helpers.to_one_of_n(
      np.arange(30) % 2)) for i in range(20)]
    m, s = perf.mean_std(perf.auc, ds)
    self.assertAlmostEqual(m, 0.5, 1)
    self.assertAlmostEqual(s, 0.1, 1)
