import unittest
import os
import numpy as np
import pylab
from numpy.testing import assert_equal
from .. import data, plots, DataSet, loss
from ..helpers import to_one_of_n, hard_max, roc, auc, auc_confidence, \
  mut_inf, write_csv_table, write_latex_table

class TestOneOfN(unittest.TestCase):
  def test_simple(self):
    '''Test to_one_of_n in simple use case'''
    # test construction with one class
    assert_equal(to_one_of_n([0, 0, 0, 0]), np.ones((4, 1)))
    assert_equal(to_one_of_n([1, 1, 1]), np.ones((3, 1)))

    # test construction with two classes, cols sorted
    ys2d_a = to_one_of_n([0, 1, 1])
    ys2d_b = to_one_of_n([1, 2, 0])
    assert_equal(ys2d_a, np.array([[1, 0], [0, 1], [0, 1]]))
    assert_equal(ys2d_b, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))

  def test_cols(self):
    '''Test to_one_of_n using given column order'''
    ys = to_one_of_n([0, 1, 2], [2, 1, 0])
    assert_equal(ys, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

  def test_cols_non_existing(self):
    '''Test to_one_of_n using non-existing column order'''
    ys = to_one_of_n([0, 1, 2], [5, 6, 7])
    np.testing.assert_equal(to_one_of_n([0, 1, 2], [5, 6, 7]), np.zeros((3, 3)))

  def test_2d_input(self):
    '''Test to_one_of_n with 2D input'''
    ok_2d = to_one_of_n(np.atleast_2d([0, 1, 1]))
    assert_equal(ok_2d, np.array([[1, 0], [0, 1], [0, 1]]))
    self.assertRaises(ValueError, to_one_of_n, np.ones((3, 3)))

class TestHardMax(unittest.TestCase):
  def test_hardmax(self):
    soft_votes = np.array([[-.3, -.1], [9, 4], [.1, .101]])
    np.testing.assert_equal(hard_max(soft_votes), to_one_of_n([1, 0, 1]))

    tie_votes = np.array([[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_equal(hard_max(tie_votes), to_one_of_n([0, 0, 0], [0, 1]))

class TestROC(unittest.TestCase):
  def test_roc(self):
    '''Test bounds and ordering of ROC'''
    TPs, FPs = roc(np.random.rand(100), np.random.rand(100).round())
    # test mononely increasing TPs and FPs
    np.testing.assert_equal(np.sort(TPs), TPs)
    np.testing.assert_equal(np.sort(FPs), FPs)
    
    self.assertEqual(TPs.min(), 0)
    self.assertEqual(TPs.max(), 1)
    self.assertEqual(FPs.min(), 0)
    self.assertEqual(FPs.max(), 1)

  def test_reverse(self):
    '''Test that the ROC is invariant for reversions'''
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)
    t1, f1 = roc(scores[::-1], labels[::-1]) # reversed ROC
    np.testing.assert_equal(t0, t1)
    np.testing.assert_equal(f0, f1)
  
  def test_known(self):
    '''Test ROC for known input'''
    scores = np.array([-1, 0, 0, 0, 0, 0, 0, 1])
    labels = np.array([ 0, 0, 0, 0, 1, 1, 1, 1])
    t0, f0 = roc(scores, labels)

    self.assert_((t0 == [0, .25, 1, 1]).all())
    self.assert_((f0 == [0, 0, .75, 1]).all())

  def test_plot(self):
    '''Test plotting ROC'''
    ys=to_one_of_n(np.linspace(0, 1, 100).round())
    d = DataSet(xs=(ys + np.random.randn(100, 2)).round(1), ys=ys)

    pylab.clf()
    plots.plot_roc(d)
    pylab.title('AUC = %.2f' % loss.auc(d))
    pylab.savefig(os.path.join('out', 'roc.eps'))
    pylab.close()

class TestAUC(unittest.TestCase):
  def test_AUC_extrema(self):
    '''Test AUC for extrema'''
    self.assertEqual(auc([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]), 1)
    self.assertEqual(auc([1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]), 0)
    self.assertEqual(auc([1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0]), .5)

  def test_AUC_symbounds(self):
    '''Test AUC for symmetry and bounds'''
    N = 100
    for rho in [.1, .3, .5]:
      for i in range(20):
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertAlmostEqual(auc(xs, ys), 1-auc(xs, np.abs(ys-1)))
        self.assert_(0 <= auc(xs, ys) <= 1)

  def test_AUC_confidence(self):
    '''Test AUC confidence interval for trends'''
    # we do not know much, but we can test for trends
    self.assert_(auc_confidence(1) > auc_confidence(100))
    self.assert_(auc_confidence(100, rho=.1) > auc_confidence(100))
    self.assert_(auc_confidence(100, delta=1e-8) > auc_confidence(100))

    # and symmetry
    for rho in [.01, .1, .5]:
      self.assertAlmostEqual(auc_confidence(100, rho=rho),
        auc_confidence(100, rho=1-rho))

  def test_monte_carlo(self):
    '''Monte Carlo test for AUC confidence intervals'''
    SAMPLES = 100
    for N in [100, 1000]:
      for rho in [0.1, .5, .9]:
        xs = np.random.random(N)
        ys = (np.linspace(0, 1, N) <= rho).round()
        self.assertEqual(ys.mean(), rho)
        aucs = []

        # create random AUCs
        for i in range(SAMPLES):
          np.random.shuffle(ys)
          aucs.append(auc(xs, ys))

        # test conservativeness
        for delta in [.95, .05, .001, .0001]:
          dev = np.abs(np.array(aucs) - 0.5)
          epsilon = auc_confidence(N, rho, delta)
          e_epsilon = sorted(dev)[int(SAMPLES * (1-delta))]
          e_p = np.mean(dev > epsilon)
          self.assert_(e_p <= delta, 
            'empirical p (=%f) > delta (=%f)' % (e_p, delta))

class TestMutualInformation(unittest.TestCase):
  def test_max_bits(self):
    for i in range(4):
      conf = np.eye(2 ** i)
      self.assertAlmostEqual(mut_inf(conf), i)

  def test_uniform(self):
    for i in range(4):
      conf = np.ones((i, i + 1))
      self.assertAlmostEqual(mut_inf(conf), 0)

  def test_zero(self):
    self.assert_(np.isnan(mut_inf(np.zeros((5, 3)))))

  def test_no_modification(self):
    conf = np.ones((4, 3))
    mut_inf(conf)
    np.testing.assert_equal(conf, np.ones((4, 3)))

  def test_symmetrical(self):
    for i in range(4):
      conf = np.random.rand(3, 8)
      self.assertAlmostEqual(mut_inf(conf), mut_inf(conf.T))

  def test_malformed(self):
    self.assertRaises(AssertionError, mut_inf, -np.ones((3, 3)))
    

class TestTables(unittest.TestCase):
  def setUp(self):
    self.table = [['H1', 'H2', 'H3'],
      ['row1,1', 'row2,2', 'row3,3'],
      ['S0', '10.1 (3.2)', '11.0 (2.9)'],
      [1, 1.3, -6]]

  def test_csv(self):
    write_csv_table(self.table, os.path.join('out', 'table.csv'))

  def test_latex(self):
    write_latex_table(self.table, os.path.join('out', 'table.tex'))
