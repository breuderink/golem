import logging
import numpy as np
from ..dataset import DataSet
from .. import helpers
from ..stat import auc
from basenode import BaseNode

class FeatFilter(BaseNode):
  def __init__(self, statistic, min_nfeatures=0, threshold=-np.inf):
    '''
    Construct a feature filter that keeps min_nfeatures, or all featurs
    that score higher than treshold using the statistic function.
    '''
    BaseNode.__init__(self)
    self.statf = statistic
    self.min_nfeatures = min_nfeatures
    self.threshold = threshold

  def train_(self, d):
    stats = np.apply_along_axis(self.statf, 1, d.X, d.Y)
    order = np.argsort(stats)[::-1]

    nfeats = max(self.min_nfeatures, np.sum(stats >= self.threshold))

    self.keep = order[:nfeats]

    self.log.info('Keeping %d%% of the features: %s' % 
      (100 * self.keep.size / d.nfeatures, self.keep))

  def apply_(self, d):
    keep = self.keep
    return DataSet(X=d.X[keep], 
      feat_lab=[d.feat_lab[i] for i in keep] if d.feat_lab != None else None,
      default=d)

  def __str__(self):
    return '%s (%susing statistic "%s")' % (self.name, 
      '%d features ' % self.keep.size if hasattr(self, 'keep') else '', 
      self.statf.__name__)

def auc_dev(X, Y):
    assert Y.shape[0] == 2, 'Use AUC with two classes.'
    aucs = auc(X.T, helpers.hard_max(Y)[1,:])
    return np.abs(aucs - .5)

class AUCFilter(FeatFilter, BaseNode):
  def __init__(self, min_auc=.6, min_nfeatures=0):
    FeatFilter.__init__(self, auc_dev, threshold=min_auc-.5, 
      min_nfeatures=min_nfeatures)
