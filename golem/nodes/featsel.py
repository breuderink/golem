import logging
import numpy as np
from ..dataset import DataSet
from .. import helpers
from basenode import BaseNode

class FeatFilter(BaseNode):
  def __init__(self, statistic, min_nfeatures=0, threshold=-np.inf):
    BaseNode.__init__(self)
    self.statf = statistic
    self.min_nfeatures = min_nfeatures
    self.threshold = threshold

  def train_(self, d):
    stats = np.apply_along_axis(self.statf, 0, d.xs, d.ys)
    order = np.argsort(stats)[::-1]

    nfeats = max(self.min_nfeatures, np.sum(stats >= self.threshold))

    self.keep = order[:nfeats]

    self.log.info('Keeping %d%% of the features: %s' % 
      (100 * self.keep.size / d.nfeatures, self.keep))

  def test_(self, d):
    keep = self.keep
    return DataSet(xs=d.xs[:, keep], 
      feat_lab=[d.feat_lab[i] for i in keep] if d.feat_lab != None else None,
      default=d)

  def __str__(self):
    return '%s (%susing statistic "%s")' % (self.name, 
      '%d features ' % self.keep.size if hasattr(self, 'keep') else '', 
      self.statf.__name__)

def auc_dev(xs, ys):
    assert ys.shape[1] == 2, 'Use AUC with two classes.'
    aucs = helpers.auc(xs, ys[:, 1] - ys[:, 0])
    return np.abs(aucs - .5)

class AUCFilter(FeatFilter, BaseNode):
  def __init__(self, min_auc=.6, min_nfeatures=0):
    FeatFilter.__init__(self, auc_dev, threshold=min_auc - .5, 
      min_nfeatures=min_nfeatures)
