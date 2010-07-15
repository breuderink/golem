import numpy as np
from dataset import DataSet
import helpers

def class_loss(d):
  '''
  Calculate discrete class loss for every instance in d.
  The resulting array contains True for instance correctly classified,
  False otherwise.'''
  return np.any(helpers.hard_max(d.xs) != helpers.hard_max(d.ys), 
    axis=1).reshape(-1, 1)

def accuracy(d):
  '''
  Return the accuracy (percentage correct) of the predictions in d.xs compared
  to the discrete class labels in d.ys.
  '''
  return 1 - np.mean(class_loss(d))

def conf_mat(d):
  '''
  Make a confusion matrix. Rows contain the label, columns the predictions.
  '''
  return np.dot(helpers.hard_max(d.ys).T, helpers.hard_max(d.xs))

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
  return helpers.auc(d.xs[:, 1] - d.xs[:, 0], helpers.hard_max(d.ys)[:,1])

def mean_std(loss_f, ds):
  '''Calc mean and std for loss function loss_f over a list with DataSets ds'''
  losses = map(loss_f, ds)
  return (np.mean(losses, axis=0), np.std(losses, axis=0))

def I(d):
  '''Mutual information'''
  return helpers.mut_inf(conf_mat(d))
