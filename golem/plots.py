import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
import helpers

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'b']

def scatter_plot(dataset):
  ''' Display the dataset with a scatterplot using Matplotlib/Pylab. The user is
  responsible for calling plt.show() to display the plot.
  '''
  assert dataset.nfeatures == 2
  scatters = []
  # loop over classes
  for ci in range(dataset.nclasses):
    color, marker = colors[ci], markers[ci]
    xs = dataset.get_class(ci).xs

    # plot features
    f0 = [x[0] for x in xs]
    f1 = [x[1] for x in xs]
    scatters.append(plt.scatter(f0, f1, s=10, c=color, marker=marker))

  assert dataset.cl_lab != []
  plt.legend(scatters, dataset.cl_lab)

  if dataset.feat_lab != None:
    xlab, ylab = dataset.feat_lab
  else:
    xlab, ylab = 'feat0', 'feat1'
  plt.xlabel(xlab)
  plt.ylabel(ylab)


def classifier_grid(classifier):
  RESOLUTION = 40
  xlim = plt.xlim()
  ylim = plt.ylim()

  # Build grid
  x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/RESOLUTION)
  y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/RESOLUTION)
  X, Y = np.meshgrid(x, y)
  xs = np.array([X.flatten(), Y.flatten()]).T

  # Get scores
  dxy = DataSet(xs=xs, ys=np.zeros((xs.shape[0], classifier.nclasses)))
  dz = classifier.test(dxy)
  Zs = []
  for ci in range(dz.nfeatures):
    pt = dz.xs[:, ci]
    prest = np.vstack([dz.xs[:, i] for i in range(dz.nfeatures) if i != ci]).T
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
    plt.contour(X, Y, Z, [0, .5, 1], linewidths=[2, .5, .5], colors='k')
  if heat_map:
    if len(Zs) > 2: 
      raise ValueError, 'Cannot draw a heat map for nclasses > 2'
    plt.imshow(Z, origin='lower', cmap=plt.cm.RdBu_r, alpha=heat_map_alpha, 
      aspect='auto', extent=[X.min(), X.max(), Y.min(), Y.max()])

def plot_roc(d, fname=None):
  '''
  Plot the ROC curve for a DataSet d. The diff of d.xs and the diff of d.ys
  is used to calculate the ranking
  '''
  assert d.nclasses == 2
  assert d.nfeatures == 2
  TPs, FPs = helpers.roc(d.xs[:, 1] - d.xs[:, 0], d.ys[:, 1] - d.ys[:, 0])
  plt.plot(FPs, TPs)
  a = plt.gca()
  a.set_aspect('equal')
  plt.axis([0, 1, 0, 1])
  plt.grid()
  plt.xlabel('False positives')
  plt.ylabel('True positives')
