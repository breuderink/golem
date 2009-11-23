import logging
import numpy as np
from ..dataset import DataSet
from .. import helpers
from basenode import BaseNode

class FeatStats(BaseNode):
  def __init__(self, statistic):
    BaseNode.__init__(self)
    self.statistic = statistic

  def train_(self, d):
    stats = [self.statistic(d.xs[:,fi], d.ys) for fi in xrange(d.nfeatures)]
    self.feat_stats = np.asarray(stats).reshape(d.feat_shape)

  def test_(self, d):
    BaseNode.test(self, d)
    return d


class FeatFilter(FeatStats):
  def __init__(self, statistic, min_nfeatures=0, threshold=0):
    FeatStats.__init__(self, statistic)
    self.min_nfeatures = min_nfeatures
    self.threshold = threshold

  def train_(self, d):
    FeatStats.train_(self, d)
    feat_bool = np.zeros(d.nfeatures, bool)
    log = self.log

    if self.min_nfeatures > 0:
      feat_bool[np.argsort(self.feat_stats.flatten())[
        -self.min_nfeatures:]] = True
      log.debug('Selector after min_nfeatures: %s.' % str(
        feat_bool.astype(int)))

    feat_bool[self.feat_stats.flatten() >= self.threshold] = True
    log.debug('Selector after thresholding: %s.' % str(feat_bool.astype(int)))

    log.info('Keeping %d%% of the features.' % (100 * feat_bool.mean()))
    if feat_bool.size <= 50:
      log.info('Keeping features %s.' % str(feat_bool.astype(int)))

    self.feat_bool = feat_bool

  def test_(self, d):
    if d.feat_lab != None:
      # Create new feature labels
      feat_lab = [d.feat_lab[i] for i in range(len(d.feat_lab)) \
        if self.feat_bool[i]]
    else:
      feat_lab = None
    return DataSet(xs=d.xs[:, self.feat_bool], feat_lab=feat_lab, default=d)

  def __str__(self):
    if hasattr(self, 'feat_bool'):
      feat_bool = self.feat_bool
      return '%s (%d of %d features using statistic "%s")' % (self.name, 
        np.sum(feat_bool), len(feat_bool), self.statistic.__name__)
    else:
      return '%s (using statatistic "%s")' % (self.name, 
        self.statistic.__name__)


def onesided_auc(xs, ys):
    assert ys.shape[1] == 2, 'Use AUC with two classes.'
    aucs = helpers.auc(xs, ys[:, 1] - ys[:, 0])
    return np.abs(aucs - .5) + .5

class AUCFilter(FeatFilter, BaseNode):
  def __init__(self, min_auc=.6, min_nfeatures=0):
    FeatFilter.__init__(self, onesided_auc, threshold=min_auc, 
      min_nfeatures=min_nfeatures)
