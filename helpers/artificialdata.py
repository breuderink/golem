import scipy as sp
import scipy.linalg as la
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

  for y in range(len(ninstances)):
    cl_instances = ninstances[y]
    xs = sp.random.multivariate_normal(mus[y], sigmas[y], cl_instances)
    ys = y * sp.ones((cl_instances, 1), int)
    d = dataset.DataSet(xs, ys) 
    result += d
  return result

