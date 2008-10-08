import numpy as np
from helpers import hard_max
from dataset import *


def class_loss(dataset):
  hm = hard_max(dataset.xs)
  ys = dataset.ys
  loss = np.where(np.sum(np.abs(hm - ys), axis=1).reshape(ys.shape[0], 1), 
    np.ones((ys.shape[0], 1)), np.zeros((ys.shape[0], 1)))
  return loss

def accuracy(dataset):
  return 1 - np.mean(class_loss(dataset))

def confusion_matrix(dataset):
  '''Make a confusion matrix. Rows contain the label, columns the prediction.'''
  result = []
  hmd = DataSet(hard_max(dataset.xs), dataset.ys)
  for ci in range(dataset.nclasses):
    cid = hmd.get_class(ci)
    result.append(np.sum(cid.xs, axis=0))
  return np.array(result).astype(int)

def format_confmat(dataset):
  c = confusion_matrix(dataset)
  formatter = lambda items : '%10s' % items[0][:10] + '|' + \
    ''.join(['%8s' % i[:8] for i in items[1:]]) 
  labels = [label[:6] for label in dataset.class_labels]
  hline = '-' * len(formatter([''] + labels))
  result = []
  
  result.append(hline)
  result.append(formatter(['True\Pred.'] + labels))
  result.append(hline)
  for ri in range(c.shape[0]):
    items = [labels[ri]] + [str(v) for v in c[ri, :]]
    result.append(formatter(items))
    #result.append(labels[ri] + '\t' + '\t'.join([str(v) for v in c[ri, :]]))
  result.append(hline)

  return '\n'.join(result)
