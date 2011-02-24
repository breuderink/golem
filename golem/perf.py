import numpy as np
from dataset import DataSet
import helpers
import stat

def class_loss(d):
  '''
  Calculate discrete class loss for every instance in d.
  The resulting array contains True for instance correctly classified,
  False otherwise.'''
  return np.any(helpers.hard_max(d.X) != helpers.hard_max(d.Y), axis=0)

def accuracy(d):
  '''
  Return the accuracy (percentage correct) of the predictions in d.X compared
  to the discrete class labels in d.Y.
  '''
  return 1 - np.mean(class_loss(d))

def conf_mat(d):
  '''
  Make a confusion matrix. Rows contain the label, columns the predictions.
  '''
  return np.dot(helpers.hard_max(d.Y), helpers.hard_max(d.X).T)

def format_confmat(conf_mat, d):
  '''
  Formats a confusion matrix conf_mat using class_labels found in DataSet d.
  '''
  conf_mat = np.asarray(conf_mat)
  assert conf_mat.shape == (d.nclasses, d.nclasses)

  labels = [label[:6] for label in d.cl_lab]
  result = [['Label\Pred.'] + labels]
  for ri in range(d.nclasses):
    result.append([labels[ri]] + conf_mat[ri].tolist())
  return result

def auc(d):
  '''
  Calculate area under curve of the ROC for a dataset d. Expects the
  predictions for a two-class problem.
  '''
  assert d.nclasses == 2 and d.nfeatures == 2
  return stat.auc(d.X[1,:] - d.X[0,:], helpers.hard_max(d.Y)[1, :])

def mean_std(loss_f, ds):
  '''Calc mean and std for loss function loss_f over a list with DataSets ds'''
  losses = map(loss_f, ds)
  return (np.mean(losses, axis=0), np.std(losses, axis=0))

def I(d):
  '''Mutual information'''
  return stat.mut_inf(conf_mat(d))
