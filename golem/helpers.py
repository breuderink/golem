import csv, itertools
import numpy as np

def to_one_of_n(labels, class_rows=None):
  '''
  Convert a list with integers to one-of-N coding for to use in a DataSet.
  Note that the rows correspond to the classes in *sorted* order.
  '''
  a = np.asarray(labels, int)
  if a.ndim != 1:
    raise ValueError('Labels should be 1D')
  if not class_rows:
    class_rows = np.unique(a) # is automatically sorted
  Y = np.zeros((len(class_rows), a.size))
  for i, n in enumerate(class_rows):
    Y[i, a==n] = 1
  return Y

def hard_max(xs):
  '''
  Find the maximum of each row and return an array containing 1 on the
  location of each maximum.
  '''
  result = np.zeros(xs.shape)
  result[range(xs.shape[0]), np.argmax(xs, axis=1)] = 1
  nans = np.where(np.any(np.isnan(xs), axis=1).reshape(-1, 1),
    np.ones(xs.shape) * np.nan, np.zeros(xs.shape))
  return result + nans


def write_csv_table(rows, fname):
  f = open(fname, 'w')
  csv.writer(f).writerows(rows)
  f.close()

def write_latex_table(rows, fname):
  rows = list(rows)
  ncols = max(len(r) for r in rows)
  f = open(fname, 'w')
  f.write('\\begin{tabular}{%s}\n' % ' '.join('c'*ncols))
  for r in rows:
    f.write(' & '.join(map(str, r)) + '\\\\\n')
  f.write('\\end{tabular}\n')
  f.close()
