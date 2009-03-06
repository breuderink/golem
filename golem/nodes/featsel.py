import logging
import numpy as np
from ..dataset import DataSet
from .. import helpers

auc_log = logging.getLogger('AUCFilter')
class AUCFilter:
  def __init__(self, min_auc=.6, min_nfeatures=0):
    self.min_auc = min_auc
    self.min_nfeatures = min_nfeatures

  def train(self, d):
    assert(d.nclasses == 2)
    labels = d.ys[:, 0]
    min_auc = self.min_auc
    aucs = np.zeros(d.nfeatures)
    for fi in xrange(d.nfeatures):
      scores = d.xs[:, fi]
      aucs[fi] = helpers.auc(scores, labels)
    auc_log.debug('Found AUCs for features: %s' % aucs)

    # select at least min_features
    self.feat_bool = np.zeros(aucs.size, bool)
    self.feat_bool[aucs.argsort()[:self.min_nfeatures]] = True

    # add threshold filtered features
    self.feat_bool[np.logical_or(aucs > min_auc, aucs < 1 - min_auc)] = True

    auc_log.info('Keeping %d%% of the features' % (100 * self.feat_bool.mean()))
    self.cl_lab = d.cl_lab
    self.aucs = aucs

  def test(self, d):
    if d.feat_lab != None:
      feat_lab = [d.feat_lab[i] for i in range(len(d.feat_lab)) \
        if self.feat_bool[i]]
    else:
      feat_lab = None
    return DataSet(xs=d.xs[:, self.feat_bool], feat_lab=feat_lab, default=d)
