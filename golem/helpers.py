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

def hard_max(X):
  '''
  Find the maximum of each column and return an array containing 1 on the
  location of each maximum. If a column contains a NaN, the output column
  consists of NaNs.
  '''
  X = np.atleast_2d(X)
  assert X.shape[0] != 0
  if X.shape[1] == 0: 
    return X.copy()
  result = np.zeros(X.shape)
  result[np.argmax(X, axis=0),range(X.shape[1])] = 1
  result[:, np.any(np.isnan(X), axis=0)] *= np.nan
  return result


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
