import numpy as np
import numpy.linalg as la
import dataset

def gaussian_dataset(ninstances = [50, 50]):
  mus = [\
    [0, 0], 
    [2, 1],
    [5, 6]]
  sigmas = [\
    [[1, 2], [2, 5]],
    [[1, 2], [2, 5]],
    [[1, -1], [-1, 2]]]

  result = dataset.DataSet()

  nclasses = len(ninstances)
  for y in range(nclasses):
    cl_instances = ninstances[y]
    xs = np.random.multivariate_normal(mus[y], sigmas[y], cl_instances)
    ys = np.zeros((cl_instances, nclasses))
    ys[:, y] = 1.
    cids = range(result.ninstances, result.ninstances + cl_instances)
    d = dataset.DataSet(xs, ys, ids=cids)
    result += d
  return result
