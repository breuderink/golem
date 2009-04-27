import unittest
import numpy as np
from .. import DataSet, loss, helpers

class TestLoss(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(
      xs=helpers.to_one_of_n([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]),
      ys=helpers.to_one_of_n([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))

  def test_hardmax(self):
    np.testing.assert_equal(helpers.hard_max(self.d.xs), self.d.xs)

    soft_votes = np.array([[-.3, -.1], [9, 4], [.1, .101]])
    np.testing.assert_equal(helpers.hard_max(soft_votes), 
      helpers.to_one_of_n([1, 0, 1]))

    tie_votes = np.array([[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_equal(helpers.hard_max(tie_votes),  
      helpers.to_one_of_n([0, 0, 0], [0, 1]))
  
  def test_class_loss(self):
    targets = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1)
    d = self.d
    d2 = DataSet(xs=d.xs + (0.1 * np.random.rand(*d.xs.shape)), default=d)
    np.testing.assert_equal(loss.class_loss(d), targets)
    np.testing.assert_equal(loss.class_loss(d2), targets)
  
  def test_accurracy(self):
    self.assertEqual(loss.accuracy(self.d), (12 - 3) / 12.)

  def testConfusionMatrix(self):
    c = loss.confusion_matrix(self.d)
    ct = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]])
    np.testing.assert_equal(c, ct)
  
  def testFormatConfusionMatrix(self):
    c = loss.format_confmat(self.d)
    target = \
      '-----------------------------------\n' + \
      'True\\Pred.|  class0  class1  class2\n' + \
      '-----------------------------------\n' + \
      '    class0|       3       1       0\n'+ \
      '    class1|       0       3       1\n' + \
      '    class2|       1       0       3\n' + \
      '-----------------------------------'
    self.assertEqual(c, target)
  
  def testAUC(self):
    d1 = DataSet(helpers.to_one_of_n([0, 0, 1, 1, 1, 1]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    d0 = DataSet(helpers.to_one_of_n([1, 1, 0, 0, 0, 0]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    dr = DataSet(helpers.to_one_of_n([1, 0, 1, 0, 1, 0]),
      helpers.to_one_of_n([1, 1, 1, 1, 0, 0]))

    self.assertEqual(loss.auc(d0), 0)
    self.assertEqual(loss.auc(d1), 1)
    self.assertEqual(loss.auc(dr), .5)
