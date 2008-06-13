import scipy as sp
import scipy.linalg as la
import dataset

def gaussian_dataset(nclasses = 2, ninstances = 50):
  mu = [0, 0]
  sigma = [[1, 2], [2, 5]]
  result = dataset.DataSet()
  for y in range(nclasses):
    xs = sp.random.multivariate_normal(mu, sigma, ninstances)
    ys = y * sp.ones((ninstances, 1))
    d = dataset.DataSet(xs, ys) 
    result += d
  return result

