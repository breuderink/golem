import numpy as np

def to_one_of_n(ints, class_cols=None):
  '''Convert a list with ints to one-of-N coding for to use in a DataSet.
  Note that the columns correspond to the classes in *sorted* order.
  '''
  a = np.array(ints, int).squeeze()
  if a.ndim != 1:
    raise ValueError('Labels should be 1D')
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
  ui = np.concatenate([np.diff(scores), np.array([1])]) != 0
  TPs, FPs = TPs[ui], FPs[ui]

  # add (0, 0) to ROC
  TPs = np.concatenate([np.array([0]), TPs])
  FPs = np.concatenate([np.array([0]), FPs])
  return (TPs, FPs)

def auc(scores, labels):
  assert len(np.unique(labels)) == 2
  TPs, FPs = roc(np.asarray(scores), np.asarray(labels))
  return np.trapz(TPs, FPs)

def auc_confidence(N, rho=.5, delta=.05):
  '''
  Calculate the confidence interval epsilon for the AUC statistic.
  N is the number of instances, rho is the percentage of *positive* instances,
  and delta is the confidence interval (.05):
  \epsilon = \sqrt{\frac{log\frac{2}{\delta}}{2\rho(1-\rho)N}}

  See:
  Shivani Agarwal, Thore Graepel, Ralf Herbrich, and Dan Roth. A large
  deviation bound for the area under the ROC curve. In Advances in Neural
  Information Processing Systems, volume 17, pages 9-16, 2005.
  '''
  return np.sqrt(np.log(2. / delta) / (2 * rho * (1 - rho) * N))
