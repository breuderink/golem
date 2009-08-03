import numpy as np
from dataset import DataSet
import helpers

def class_loss(d):
  hm = helpers.hard_max(d.xs)
  ys = d.ys
  loss = np.where(np.sum(np.abs(hm - ys), axis=1).reshape(-1, 1), 
    np.ones((ys.shape[0], 1)), np.zeros((ys.shape[0], 1)))
  return loss

def accuracy(d):
  return 1 - np.mean(class_loss(d))

def confusion_matrix(d):
  '''Make a confusion matrix. Rows contain the label, columns the prediction.'''
  result = []
  hmd = DataSet(helpers.hard_max(d.xs), d.ys, None)
  for ci in range(d.nclasses):
    cid = hmd.get_class(ci)
    result.append(np.sum(cid.xs, axis=0))
  return np.array(result).astype(int)

def format_confmat(d):
  c = confusion_matrix(d)
  formatter = lambda items : '%10s' % items[0][:10] + '|' + \
    ''.join(['%8s' % i[:8] for i in items[1:]]) 
  labels = [label[:6] for label in d.cl_lab]
  hline = '-' * len(formatter([''] + labels))
  result = []
  
  result.append(hline)
  result.append(formatter(['True\Pred.'] + labels))
  result.append(hline)
  for ri in range(c.shape[0]):
    items = [labels[ri]] + [str(v) for v in c[ri, :]]
    result.append(formatter(items))
  result.append(hline)

  return '\n'.join(result)

def auc(d):
  assert d.nclasses == 2 and d.nfeatures == 2
  return helpers.auc(d.xs[:, 1] - d.xs[:, 0], d.ys[:, 1] - d.ys[:, 0])

def mean_std(loss_f, ds):
  '''Calc mean and std for loss function loss_f over a list with DataSets ds'''
  losses = [loss_f(d) for d in ds]
  return (np.mean(losses, axis=0), np.std(losses, axis=0))

def I(d):
  '''Mutual information'''
  return helpers.mut_inf(confusion_matrix(d))
