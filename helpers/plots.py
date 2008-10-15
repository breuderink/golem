# TODO:
# - add markers to the legend
# - use class names if provided
import pylab
import numpy as np
from dataset import *

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'b']

def scatter_plot(dataset, fname = None):
  ''' Display the dataset with a scatterplot using Matplotlib/pylab. The user is
  responsible for calling pylab.show() to display the plot.

  '''
  assert(dataset.nfeatures == 2)
  # loop over classes
  for ci in range(dataset.nclasses):
    color, marker = colors[ci], markers[ci]
    xs = dataset.get_class(ci).xs

    # plot features
    f0 = [x[0] for x in xs]
    f1 = [x[1] for x in xs]
    pylab.scatter(f0, f1, c = color, marker = marker, 
      label = dataset.class_labels[ci])
  pylab.legend()
  pylab.xlabel(dataset.feature_labels[0])
  pylab.ylabel(dataset.feature_labels[1])
  if fname:
    pylab.savefig(fname)
    pylab.close()

def classifier_grid(classifier):
  resolution = 80
  xlim = pylab.xlim()
  ylim = pylab.ylim()

  # Build grid
  x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/resolution)
  y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/resolution)
  X, Y = np.meshgrid(x, y)
  xs = np.array([X.flatten(), Y.flatten()]).T

  # Get scores
  dxy = DataSet(xs, np.zeros((xs.shape[0], 2)), None)
  dz = classifier.test(dxy)
  Zs = []
  for ci in range(dz.nfeatures):
    pt = dz.xs[:, ci]
    prest = np.vstack([dz.xs[:, i] for i in range(dz.nfeatures) if i <> ci]).T
    Z = pt - np.max(prest, axis=1)
    Z = Z.reshape(X.shape)
    Zs.append(Z)
  return (X, Y, Zs)

def plot_classifier_hyperplane(classifier, fname=None, heat_map=False, 
  heat_map_alpha=0.8):
  '''
  Plot the decision-function of a classifier. The labels of the contours can
  be enabled with contour_label, plotting the heatmap can be disabled with the
  heat_map argument.

  '''
  (X, Y, Zs) = classifier_grid(classifier)
  for Z in Zs:
    pylab.contour(X, Y, Z, [0, .5, 1], linewidths=[2, .5, .5], colors='k')
  if heat_map:
    if len(Zs) > 2: raise ValueError, 'Cannot draw a heat map for nclasses > 2'
    pylab.imshow(Z, origin='lower', alpha=heat_map_alpha, aspect='auto',
      extent=[X.min(), X.max(), Y.min(), Y.max()])
  if fname:
    pylab.savefig(fname)
    pylab.close()
