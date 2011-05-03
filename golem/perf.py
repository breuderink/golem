import warnings
import numpy as np
from dataset import DataSet
import helpers
import stat

def class_loss(d):
  r'''
  Calculate discrete class loss for every instance in d.
  The resulting array contains 1 for instance incorrectly classified,
  0 otherwise:

  >>> d = DataSet(X=helpers.to_one_of_n([0, 0, 0, 1, 1, 2]), 
  ...             Y=helpers.to_one_of_n([0, 0, 1, 0, 1, 2]))
  >>> class_loss(d)
  array([ 0.,  0.,  1.,  1.,  0.,  0.])
  '''
  assert d.nfeatures == d.nclasses
  return np.any(
    helpers.hard_max(d.X) != helpers.hard_max(d.Y), axis=0).astype(float)

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

def format_confmat(confmat, d):
  '''
  Formats a confusion matrix confmat using class_labels found in DataSet d.
  '''
  confmat = np.asarray(confmat)
  assert confmat.shape == (d.nclasses, d.nclasses)

  labels = [label[:6] for label in d.cl_lab]
  result = [['Label\Pred.'] + labels]
  for ri in range(d.nclasses):
    result.append([labels[ri]] + confmat[ri].tolist())
  return result

def auc(d):
  '''
  Calculate area under curve of the ROC for a dataset d. Expects the
  predictions for a two-class problem.
  '''
  assert d.nclasses == 2 and d.nfeatures == 2
  return stat.auc(np.diff(d.X, axis=0)[0], helpers.hard_max(d.Y)[1])

def mean_std(loss_f, ds):
  '''
  Calculate mean and std for loss function loss_f over a list ds with DataSets
  '''
  losses = map(loss_f, ds)
  return (np.mean(losses, axis=0), np.std(losses, axis=0))

def I(d):
  '''Mutual information'''
  warnings.warn('perf.I is deprecated, use perf.mutinf instead.', 
      DeprecationWarning)
  return mutinf(d)

def mutinf(d):
  '''Calculate mutual information based on the confusion matrix.'''
  return stat.mut_inf(conf_mat(d))
