import numpy as np
import numpy.linalg as la
from dataset import DataSet
from helpers import to_one_of_n

def gaussian_dataset(ninstances=[50, 50]):
  '''
  Simple Gaussian dataset with a variable number of instances for up to 3
  classes.
  '''
  mus = [\
    [0, 0], 
    [2, 1],
    [5, 6]]
  sigmas = [\
    [[1, 2], [2, 5]],
    [[1, 2], [2, 5]],
    [[1, -1], [-1, 2]]]

  assert len(ninstances) <= 3

  nclasses = len(ninstances)
  Xs, Ys = [], []

  for (ci, n) in enumerate(ninstances):
    Xs.append(np.random.multivariate_normal(mus[ci], sigmas[ci], n).T)
    Ys.extend(np.ones(n) * ci)

  return DataSet(X=np.hstack(Xs), Y=to_one_of_n(Ys))

def wieland_spirals():
  '''
  Famous non-linear binary 2D problem with intertwined spirals.
  '''
  i = np.arange(97)
  theta = np.pi * i / 16.
  r = 6.5 * (104 - i) / 104.
  X = np.array([r * np.cos(theta), r * np.sin(theta)])
  X = np.hstack([X, -X])
  Y = to_one_of_n(np.hstack([np.zeros(i.size), np.ones(i.size)]))
  return DataSet(X=X, Y=Y)
