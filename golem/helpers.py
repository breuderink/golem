import numpy as np

def to_one_of_n(ints, nclasses=None):
  '''Convert a list with ints to one-of-N coding for to use in a DataSet.
  Note that the columns correspond to the classes in *sorted* order.

  '''
  a = np.array(ints)
  classes = np.unique(a)
  if not nclasses:
    nclasses = classes.size
  ys = np.zeros((a.size, nclasses))  
  for i in range(classes.size):
    ys[a == classes[i], i] = 1
  return ys

def hard_max(xs):
  '''Find the maximum of each row and return an array containing 1 on the
  location of each maximum.
  '''
  result = np.zeros(xs.shape)
  result[range(xs.shape[0]), np.argmax(xs, axis=1)] = 1
  return result

def roc(scores, labels):
  '''Calc (TPs, FPs) for ROC plotting and AUC-ROC calculation.''' 
  assert(scores.ndim == labels.ndim ==  1)
  si = np.argsort(scores)[::-1]
  scores, labels = scores[si], labels[si]
  
  # slide threshold from above
  TPs = np.cumsum(labels == 1) / np.sum(labels == 1).astype(float)
  FPs = np.cumsum(labels <> 1) / np.sum(labels <> 1).astype(float)
  
  # handle equal scoress
  ui = np.concatenate([np.diff(scores), np.array([1])]) <> 0
  TPs, FPs = TPs[ui], FPs[ui]

  # add (0, 0) to ROC
  TPs = np.concatenate([np.array([0]), TPs])
  FPs = np.concatenate([np.array([0]), FPs])
  return (TPs, FPs)

def auc(scores, labels):
  TPs, FPs = roc(scores, labels)
  return np.trapz(TPs, FPs)
