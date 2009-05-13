import logging
import numpy as np
from ..dataset import DataSet
from .. import helpers

log = logging.getLogger('FeatFilter')
auc_log = logging.getLogger('AUCFilter')

class FeatStats:
  def __init__(self, statistic):
    self.statistic = statistic

  def train(self, d):
    stats = [self.statistic(d.xs[:,fi], d.ys) for fi in xrange(d.nfeatures)]
    self.feat_stats = np.asarray(stats).reshape(d.feat_shape)

  def test(self, d):
    return d


class FeatFilter(FeatStats):
  def __init__(self, statistic, min_nfeatures=0, threshold=0):
    FeatStats.__init__(self, statistic)
    self.min_nfeatures = min_nfeatures
    self.threshold = threshold

  def train(self, d):
    FeatStats.train(self, d)
    feat_bool = np.zeros(d.nfeatures, bool)

    if self.min_nfeatures > 0:
      feat_bool[np.argsort(self.feat_stats)[-self.min_nfeatures:]] = True
      log.debug('Selector after min_nfeatures: %s' % str(feat_bool.astype(int)))

    feat_bool[self.feat_stats >= self.threshold] = True
    log.debug('Selector after thresholding: %s' % str(feat_bool.astype(int)))

    log.info('Keeping %d%% of the features' % (100 * feat_bool.mean()))
    if feat_bool.size <= 50:
      log.info('Keeping features %s' % str(feat_bool.astype(int)))

    self.feat_bool = feat_bool

  def test(self, d):
    if d.feat_lab != None:
      # Create new feature labels
      feat_lab = [d.feat_lab[i] for i in range(len(d.feat_lab)) \
        if self.feat_bool[i]]
    else:
      feat_lab = None
    return DataSet(xs=d.xs[:, self.feat_bool], feat_lab=feat_lab, default=d)

def pos_auc_filter(xs, ys):
    assert ys.shape[1] == 2, 'Cannot use AUC for > 2 classes'
    aucs = helpers.auc(xs, ys[:,-1])
    return np.abs(aucs - .5) + .5

class AUCFilter(FeatFilter):
  def __init__(self, min_auc=.6, min_nfeatures=0):
    FeatFilter.__init__(self, pos_auc_filter, threshold=min_auc, 
      min_nfeatures=min_nfeatures)
