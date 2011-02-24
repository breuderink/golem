import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
import helpers

def plot_classifier(classifier, d, densities=True, log_p=False):
  '''
  High level function to plot a 2D classifier. This function 
  1) makes a scatterplot of d
  2) draws the hyperplane(s) of the classifier
  3) optionally, draws the postior probability mass.
  
  When log_p is true, the desities are assumed to be log-transformed,
  as is usually the case with the LDA etc.
  '''
  RESOLUTION = 80
  # Manually calculate limits
  center = np.mean(d.xs, axis=0)
  max_dev = np.std(np.abs(d.X.T-center), axis=0) * 5
  lims = np.array([center - max_dev, center + max_dev])
  xlim, ylim = lims[:, 0], lims[:, 1]

  # Evaluate on grid.
  (X, Y, Zs) = classifier_grid(classifier, 
    np.linspace(xlim[0], xlim[1], RESOLUTION), 
    np.linspace(ylim[0], ylim[1], RESOLUTION))
  plot_hyperplane((X, Y, Zs))
  if densities:
    plot_densities((X, Y, np.exp(Zs) if log_p else Zs))
  scatter_plot(d) 
  plt.title(str(classifier))
  plt.xlim(xlim)
  plt.ylim(ylim)

def scatter_plot(d):
  '''
  Display the d with a scatterplot using Matplotlib/Pylab. The user is
  responsible for calling plt.show() to display the plot.
  '''
  MARKERS = ['o', 'o', 's', 'd', 'v']
  COLORS = ['w', 'k', 'r', 'y', 'b']
  assert d.nfeatures == 2, 'Can only scatter a DataSet with 2 features.'
  scatters = []
  # loop over classes
  for ci in range(d.nclasses):
    color, marker = COLORS[ci], MARKERS[ci]
    X = d.get_class(ci).X

    # plot features
    scatters.append(plt.scatter(X[0], X[1], s=15, c=color, marker=marker))

  assert d.cl_lab != []
  plt.legend(scatters, d.cl_lab)

  if d.feat_lab != None:
    xlab, ylab = d.feat_lab
  else:
    xlab, ylab = 'feat0', 'feat1'
  plt.xlabel(xlab)
  plt.ylabel(ylab)

def classifier_grid(classifier, x, y):
  '''
  Evaluate a classifier at a 2D grid, specified by coordinates x and y.
  Used for hyperplane and density plots.

  Returns (X, Y, Zs), where X contains all x-coords, Y contains all y-coords,
  and Zs is [nx x ny x nclasses] array.
  '''
  # build grid
  A, B = np.meshgrid(x, y)
  X = np.array([A.flatten(), B.flatten()])
  d = DataSet(X=X, Y=np.zeros((classifier.nclasses, X.shape[1])))

  # get scores
  preds = classifier.apply(d)
  Zs = preds.X.reshape(d.nclasses, A.shape[0], A.shape[1])
  Zs = np.rollaxis(Zs, 0, 3)
  return (A, B, Zs)

def plot_hyperplane((X, Y, Zs)):
  '''
  Plots the hyperplane(s) of a classifier, given the results of classifier_grid.
  '''
  for ci in range(Zs.shape[-1]):
    # subtract competing class to get nice sharp boundaries
    Z = Zs[:,:,ci] - np.max(Zs[:,:,np.arange(Zs.shape[2]) != ci], axis=2)
    plt.contour(X, Y, Z, [0], linewidths=2, colors='k')

def plot_densities((X, Y, Zs)):
  '''
  Plots the iso-lines for the densities, given the results of classifier_grid.
  '''
  # p < 0 is nonsense. Using 0 prevents to dense graphs for for example the SVM
  heights = np.linspace(0, np.max(Zs), 7) 
  for ci in range(Zs.shape[-1]):
    Z = Zs[:, :, ci]
    cs = plt.contour(X, Y, Z, heights, linewidths=.3, colors='k')
    plt.clabel(cs, fontsize=6)

def plot_roc(d, fname=None):
  '''
  Plot the ROC curve for a DataSet d. The diff of d.X and hard_max(d.Y[1])
  is used to calculate the ranking
  '''
  assert d.nclasses == 2
  assert d.nfeatures == 2
  TPs, FPs = helpers.roc(np.diff(d.X, axis=0)[0], helpers.hard_max(d.Y)[1])
  plt.plot(FPs, TPs)
  a = plt.gca()
  a.set_aspect('equal')
  plt.axis([0, 1, 0, 1])
  plt.grid(True)
  plt.xlabel('False positives')
  plt.ylabel('True positives')
