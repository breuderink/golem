import scipy as sp
import scipy.linalg as la


import sys
sys.path.append('..')
import dataset

def gaussian_dataset(mu, sigma, nclasses = 2, ninstances = 50):
  result = dataset.DataSet()
  for y in range(nclasses):
    xs = sp.random.multivariate_normal(mu, sigma, ninstances)
    ys = y * sp.ones((ninstances, 1))
    d = dataset.DataSet(xs, ys) 
    result += d
  return result

mu = [0, 0]
sigma = [[1, 2], [2, 5]]

print sigma, sp.array(sigma)
gaussian_dataset(mu, sigma, ninstances = 3)
