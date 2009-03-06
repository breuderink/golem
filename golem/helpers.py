import numpy as np

def to_one_of_n(ints, class_cols=None):
  '''Convert a list with ints to one-of-N coding for to use in a DataSet.
  Note that the columns correspond to the classes in *sorted* order.
  '''
  a = np.array(ints, int)
  assert a.ndim == 1, 'Labels should be 1D'
  if not class_cols:
    class_cols = np.unique(a) # is automatically sorted
  ys = np.zeros((a.size, len(class_cols)))
  for i in range(len(class_cols)):
    ys[a == class_cols[i], i] = 1
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
