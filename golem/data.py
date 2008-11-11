import numpy as np
import numpy.linalg as la
from golem import DataSet

def gaussian_dataset(ninstances = [50, 50]):
  mus = [\
    [0, 0], 
    [2, 1],
    [5, 6]]
  sigmas = [\
    [[1, 2], [2, 5]],
    [[1, 2], [2, 5]],
    [[1, -1], [-1, 2]]]

  result = []
  nclasses = len(ninstances)
  last_id = 0
  for y in range(nclasses):
    cl_instances = ninstances[y]
    xs = np.random.multivariate_normal(mus[y], sigmas[y], cl_instances)
    ys = np.zeros((cl_instances, nclasses))
    ys[:, y] = 1.
    result.append(DataSet(xs, ys, range(last_id, last_id + cl_instances)))
    last_id += cl_instances 
  result = reduce(lambda a, b: a + b, result)
  return result
