
import random
import sys
sys.path.append('..')
import dataset

d = dataset.DataSet()
for y in range(5):
  for i in range(30):
    d.add_instance([random.random() for x in range(2)], y)
scatterplot(d)
scatterplot(d, 'test.svg')
scatterplot(d, 'test.eps')
scatterplot(d, 'test.png')

pylab.show()
