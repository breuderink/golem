import numpy as np
import numpy.linalg as la
from dataset import DataSet
from helpers import to_one_of_n

def gaussian_dataset(ninstances=[50, 50]):
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
