# TODO:
# - add markers to the legend
# - use class names if provided
import pylab
import numpy as np

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'b']

def scatter_plot(dataset, fname = None):
  ''' Display the dataset with a scatterplot using Matplotlib/pylab. The user is
  responsible for calling pylab.show() to display the plot.

  '''
  assert(dataset.nfeatures == 2)
  labels = dataset.labels 
  # loop over classes
  for ci in range(len(labels)):
    color, marker = colors[ci], markers[ci]

    # extract relevant instances
    xs = [x for (x, y) in dataset if y == labels[ci]]

    # plot features
    f0 = [x[0] for x in xs]
    f1 = [x[1] for x in xs]
    pylab.scatter(f0, f1, c = color, marker = marker, 
      label = dataset.class_label(ci))
  pylab.legend()
  pylab.xlabel(dataset.feature_label(0))
  pylab.ylabel(dataset.feature_label(1))
  if fname:
    pylab.savefig(fname)
    pylab.close()

def classifier_grid(classifier):
  # add contours
  resolution = 50
  xlim = pylab.xlim()
  ylim = pylab.ylim()

  x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/resolution)
  y = np.arange(ylim[0], ylim[1], (ylim[1]-ylim[0])/resolution)
  X, Y = np.meshgrid(x, y)
  xs = np.array([X.flatten(), Y.flatten()]).T
  #Z = classifier(xs).reshape(X.shape)
  ys = classifier.test(xs)
  ys = ys[:, 0] - ys[:, 1]
  Z = ys.reshape(X.shape)
  return (X, Y, Z)

def plot_classifier_hyperplane(classifier, contour_label=False, heat_map=True, 
  heat_map_alpha = 0.8, fname=None):
  '''Plot the decision-function of a classifier. The labels of the contours can
  be enabled with contour_label, plotting the heatmap can be disabled with the
  heat_map argument.

  '''
  (X, Y, Z) = classifier_grid(classifier)
  contour = pylab.contour(X, Y, Z, [-1, 0, 1], linewidths=[1, 2, 1], colors='k')
  if contour_label:
    pylab.clabel(contour)
  if heat_map:
    pylab.imshow(Z, origin='lower', alpha=heat_map_alpha, aspect='auto',
      extent=[X.min(), X.max(), Y.min(), Y.max()])
  if fname:
    pylab.savefig(fname)
    pylab.close()
