import numpy
from pprint import pprint
import csv

from data import Data
from label_to_vector import LabelToVector
from knn import *
from chain import *
from group import *
from loss import *

def read_num_csv(fname, ntype = float):
  '''Simple reader that reads a CSV file and returns a list of ntype typed rows'''
  return [map(ntype, row) for row in csv.reader(open(fname, 'r'))]
  
print 'Loading...'
train_xs = read_num_csv('/home/boris/Datasets/IRIS/train_xs.csv', ntype = float)
train_ys = read_num_csv('/home/boris/Datasets/IRIS/train_ys.csv', ntype = str)

print 'Training...'
train_data = Data(train_xs, train_ys, ['CSV'])

classifiers = [Chain([KNN(K=n), Loss()]) for n in [1, 3, 5]]
tail = Group(classifiers)

system = Chain([LabelToVector(), tail])
train_result = system.train(train_data)

print 'Loading...'
test_xs = read_num_csv('/home/boris/Datasets/IRIS/test_xs.csv', ntype = float)
test_ys = read_num_csv('/home/boris/Datasets/IRIS/test_ys.csv', ntype = str)

print 'Testing...'
test_data = Data(test_xs, test_ys, ['CSV'])
result = system.test(test_data)

print result[0]
print result[1]
print result[2]

print [b.xs for b in result]
