# TODO:
# - add markers to the legend
# - use class names if provided
import pylab
import numpy

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'b']

def scatter_plot(dataset, fname = None):
  ''' Display the dataset with a scatterplot using Matplotlib/pylab. The user is
  responsible for calling pylab.show() to display the plot.

  '''
  assert(dataset.nfeatures == 2)
  labels = dataset.labels 
  pylab.figure()
  # loop over classes
  for yi in range(len(labels)):
    color, marker = colors[yi], markers[yi]

    # extract relevant instances
    xs = [x for (x, y) in dataset if y == labels[yi]]

    # plot features
    f0 = [x[0] for x in xs]
    f1 = [x[1] for x in xs]
    pylab.scatter(f0, f1, c = color, marker = marker, 
      label = dataset.class_label(yi))
  pylab.axis('equal') # otherwise the scale is hardly visible
  pylab.legend()
  pylab.xlabel(dataset.feature_label(0))
  pylab.ylabel(dataset.feature_label(1))

  if fname:
    pylab.savefig(fname)
    pylab.close()
  else:
    pylab.show()

