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
