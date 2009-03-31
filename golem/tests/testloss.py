import unittest
import numpy as np
from .. import DataSet, loss, helpers

class TestLoss(unittest.TestCase):
  def setUp(self):
    self.d = DataSet(
      helpers.to_one_of_n([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0]),
      helpers.to_one_of_n([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]), None)

  def testHardMax(self):
    self.assert_((helpers.hard_max(self.d.xs) == self.d.xs).all())

    soft_votes = np.array([[-.3, -.1], [9, 4], [.1, .101]])
    self.assert_((helpers.hard_max(soft_votes) == 
      helpers.to_one_of_n([1, 0, 1])).all())

    tie_votes = np.array([[-1, -1], [0, 0], [1, 1]])
    self.assert_((helpers.hard_max(tie_votes) == 
      helpers.to_one_of_n([0, 0, 0], [0, 1])).all())
  
  def testClassLoss(self):
    self.assert_((loss.class_loss(self.d) == 
      np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(12, 1)).all())
  
  def testAccurracy(self):
    self.assert_(loss.accuracy(self.d) == (12 - 3) / 12.)

  def testConfusionMatrix(self):
    c = loss.confusion_matrix(self.d)
    ct = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]])
    self.assert_((c == ct).all())
  
  def testFormatConfusionMatrix(self):
    c  = loss.format_confmat(self.d)
    self.assert_(hash(c) == 630198187)
  
  def testAUC(self):
    d1 = DataSet(helpers.to_one_of_n([0, 0, 1, 1, 1, 1]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    d0 = DataSet(helpers.to_one_of_n([1, 1, 0, 0, 0, 0]),
      helpers.to_one_of_n([0, 0, 1, 1, 1, 1]))
    dr = DataSet(helpers.to_one_of_n([1, 0, 1, 0, 1, 0]),
      helpers.to_one_of_n([1, 1, 1, 1, 0, 0]))

    self.assert_(loss.auc(d0) == 1)
    self.assert_(loss.auc(d1) == 0)
    self.assert_(loss.auc(dr) == .5)

    
if __name__ == '__main__':
  unittest.main()
