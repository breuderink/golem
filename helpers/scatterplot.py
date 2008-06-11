import pylab
import numpy

markers = ['o', 'o', 's', 'd', 'v']
colors = ['w', 'k', 'r', 'y', 'g']

def scatterplot(dataset, fname = None):
  '''
  Display the dataset with a scatterplot using Matplotlib/pylab. The user is
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
      label = 'class %s' % labels[yi])
  pylab.legend()
  if fname:
    pylab.savefig(fname)
    pylab.close()

import random
import sys
sys.path.append('..')
import dataset

d = dataset.DataSet()
for y in range(4):
  for i in range(3):
    d.add_instance([random.random() for x in range(2)], y)
scatterplot(d)
scatterplot(d, 'test.svg')
scatterplot(d, 'test.eps')
scatterplot(d, 'test.png')

pylab.show()
