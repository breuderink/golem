import unittest, operator
import numpy as np
from .. import cv, data, perf
from ..nodes import PriorClassifier
from .. import DataSet

class TestCrossValidation(unittest.TestCase):
  def setUp(self):
    self.d = data.gaussian_dataset([30, 20, 10])

  def check_disjoint(self, subsets):
    '''Test that subsets are disjoint datasets'''
    for (tr, te) in cv.cross_validation_sets(subsets):
      intersection = set(tr.ids.flatten()).intersection(te.ids.flatten()) 
      self.assertEqual(len(intersection), 0)

  def test_stratified_split(self):
    '''Test stratified splitting of a DataSet'''
    d = self.d

    subsets = cv.strat_splits(d, 10)
    self.assertEqual(len(subsets), 10)
    for s in subsets:
      self.assertEqual(s.ninstances_per_class, [3, 2, 1])
    
    self.check_disjoint(subsets)

    d2 = reduce(operator.add, subsets)
    self.assertEqual(d.sorted(), d2.sorted())
  
  def test_sequential_split(self):
    '''Test sequentially splitting of a DataSet'''
    d = self.d
    for K in [3, 9, 10]:
      subsets = cv.seq_splits(d, K)
      self.assertEqual(len(subsets), K)
      for s in subsets:
        self.assert_(s.ninstances >= np.floor(d.ninstances/float(K)))
        self.assert_(s.ninstances <= np.ceil(d.ninstances/float(K)))
      
      self.check_disjoint(subsets)
      d2 = reduce(operator.add, subsets)
      self.assertEqual(d.sorted(), d2.sorted())
  
  def test_crossvalidation_sets(self):
    '''Test the generation of cross-validation training and test sets'''
    subsets = cv.strat_splits(self.d, 8)
    cv_sets = [tu for tu in cv.cross_validation_sets(subsets)]
    self.assertEqual(len(cv_sets), 8)
    for (tr, te) in cv_sets:
      self.assertEqual((tr + te).sorted(), self.d.sorted()) # tr + te == d

  def test_crossvalidation_label_prot(self):
    '''Test protection against reading labels in cross-validation'''
    class CheckNode:
      def train(self, d): pass
      def apply(self, d): 
        assert(np.all(np.isnan(d.ys)))
        return d

    preds = cv.cross_validate(cv.strat_splits(self.d, 8), CheckNode())

  def test_crossvalidation_mem_prot(self):
    '''Test protection against remembering over repetitions'''
    class MemNode:
      def __init__(self):
        self.mem = {}

      def train(self, d):
        pairs = zip(d.ids.flat, d.ys.tolist())
        self.mem = dict(self.mem.items() + pairs)

      def apply(self, d): 
        xs = np.asarray([self.mem.get(i, [0, 0, 0]) for i in d.ids.flat])
        return DataSet(xs=xs, default=d)

    a = perf.mean_std(perf.accuracy,
      cv.cross_validate(cv.strat_splits(self.d, 4), MemNode()))
    self.assertAlmostEqual(a[0], .5, 1)

  def test_rep_cv(self):
    d = self.d
    c = PriorClassifier()  
    tds = list(cv.rep_cv(d, c))
    self.assertEqual(len(tds), 50)
    self.assertAlmostEqual(perf.mean_std(perf.accuracy, tds)[0], .5)
    fold_ids = np.array([td.ids.flatten() for td in tds])
    self.failIf(
      (np.var(fold_ids.reshape(-1, d.ninstances), axis=0) == 0).any(),
      'The folds are all the same!')
