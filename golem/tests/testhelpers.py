import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_equal
from .. import data, plots, DataSet, perf
from ..helpers import to_one_of_n, hard_max, write_csv_table, \
  write_latex_table

class TestOneOfN(unittest.TestCase):
  def test_simple(self):
    '''Test to_one_of_n in simple use case'''
    # test construction with one class
    assert_equal(to_one_of_n([0]), np.ones((1, 1)))
    assert_equal(to_one_of_n([0, 0, 0, 0]), np.ones((1, 4)))
    assert_equal(to_one_of_n([1, 1, 1]), np.ones((1, 3)))

    # test construction with two classes, rows sorted
    Y2d_a = to_one_of_n([0, 1, 1])
    Y2d_b = to_one_of_n([1, 2, 0])
    assert_equal(Y2d_a, np.array([[1, 0, 0], [0, 1, 1]]))
    assert_equal(Y2d_b, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

  def test_rows(self):
    '''Test to_one_of_n using given row order'''
    Y = to_one_of_n([0, 1, 2], [2, 1, 0])
    assert_equal(Y, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))

  def test_cols_non_existing(self):
    '''Test to_one_of_n using non-existing column order'''
    Y = to_one_of_n([0, 1, 2], [5, 6, 7])
    np.testing.assert_equal(Y, np.zeros((3, 3)))

  def test_2d_input(self):
    self.assertRaises(ValueError, to_one_of_n, np.ones((3, 1)))

class TestHardMax(unittest.TestCase):
  def test_hardmax(self):
    soft_votes = np.array([[-.3, -.1], [9, 4], [.1, .101]])
    np.testing.assert_equal(hard_max(soft_votes), to_one_of_n([1, 0, 1]).T)

    tie_votes = np.array([[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_equal(hard_max(tie_votes), to_one_of_n([0, 0, 0], [0, 1]).T)

  def test_nan(self):
    soft_votes = np.array([[0, 1], [1, 0], [np.nan, 0]])
    np.testing.assert_equal(hard_max(soft_votes), 
      np.asarray([[0, 1], [1, 0], [np.nan, np.nan]]))

  def test_empty(self):
    hard_max(np.zeros((0, 3)))

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
