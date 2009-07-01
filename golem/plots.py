import pylab
import numpy as np
from dataset import DataSet
import helpers

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'b']

def scatter_plot(dataset):
  ''' Display the dataset with a scatterplot using Matplotlib/pylab. The user is
  responsible for calling pylab.show() to display the plot.
  '''
  assert dataset.nfeatures == 2
  # loop over classes
  for ci in range(dataset.nclasses):
    color, marker = colors[ci], markers[ci]
    xs = dataset.get_class(ci).xs

    # plot features
    f0 = [x[0] for x in xs]
    f1 = [x[1] for x in xs]
    pylab.scatter(f0, f1, c=color, marker=marker, label=dataset.cl_lab[ci])
  pylab.legend()

  if dataset.feat_lab != None:
    xlab, ylab = dataset.feat_lab
  else:
    xlab, ylab = 'feat0', 'feat1'
  pylab.xlabel(xlab)
  pylab.ylabel(ylab)


def classifier_grid(classifier):
  RESOLUTION = 40
  xlim = pylab.xlim()
  ylim = pylab.ylim()

  # Build grid
  x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/RESOLUTION)
  y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/RESOLUTION)
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

def plot_classifier_hyperplane(classifier, heat_map=False, heat_map_alpha=1):
  '''
  Plot the decision-function of a classifier. The labels of the contours can
  be enabled with contour_label, plotting the heatmap can be disabled with the
  heat_map argument.
  '''
  (X, Y, Zs) = classifier_grid(classifier)
  for Z in Zs:
    pylab.contour(X, Y, Z, [0, .5, 1], linewidths=[2, .5, .5], colors='k')
  if heat_map:
    if len(Zs) > 2: 
      raise ValueError, 'Cannot draw a heat map for nclasses > 2'
    pylab.imshow(Z, origin='lower', cmap=pylab.cm.RdBu_r, alpha=heat_map_alpha, 
      aspect='auto', extent=[X.min(), X.max(), Y.min(), Y.max()])

def plot_roc(d, fname=None):
  '''
  Plot the ROC curve for a DataSet d. The diff of d.xs and the diff of d.ys
  is used to calculate the ranking
  '''
  assert d.nclasses == 2
  TPs, FPs = helpers.roc(np.diff(d.xs, axis=1).flat, np.diff(d.ys, axis=1).flat)
  pylab.plot(FPs, TPs)
  a = pylab.gca()
  a.set_aspect('equal')
  pylab.axis([0, 1, 0, 1])
  pylab.grid()
  pylab.xlabel('False positives')
  pylab.ylabel('True positives')
