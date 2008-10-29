__all__ = ['plots', 'artificialdata']

import numpy as np

def to_one_of_n(ints, nclasses=None):
  '''Convert a list with ints to one-of-N coding for to use in a DataSet.
  Note that the columns correspond to the classes in *sorted* order.

  '''
  a = np.array(ints)
  classes = np.unique(a)
  if not nclasses:
    nclasses = classes.size
  ys = np.zeros((a.size, nclasses))  
  for i in range(classes.size):
    ys[a == classes[i], i] = 1
  return ys

def hard_max(xs):
  '''Find the maximum of each row and return an array containing 1 on the
  location of each maximum.

  '''
  result = np.zeros(xs.shape)
  result[range(xs.shape[0]), np.argmax(xs, axis=1)] = 1
  return result

