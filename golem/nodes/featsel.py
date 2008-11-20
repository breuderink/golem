import logging
import numpy as np
from golem import DataSet, helpers

log = logging.getLogger('AUCFilter')
class AUCFilter:
  def __init__(self, min_auc = .6):
    self.min_auc = min_auc

  def train(self, d):
    assert(d.nclasses == 2)
    labels = d.ys[:, 0]
    min_auc = self.min_auc
    aucs = np.zeros(d.nfeatures)
    for fi in xrange(d.nfeatures):
      scores = d.xs[:, fi]
      aucs[fi] = helpers.auc(scores, labels)
    log.debug('Found AUCs for features: %s' % aucs)
    self.feat_bool = np.logical_or(aucs > min_auc, aucs < 1 - min_auc)
    log.info('Keeping %d%% of the features' % (100 * np.mean(self.feat_bool)))
    
    self.cl_lab = d.cl_lab
    self.aucs = aucs

  def test(self, d):
    feat_lab = [d.feat_lab[i] for i in range(len(d.feat_lab)) \
      if self.feat_bool[i]]
    print feat_lab
    return DataSet(xs=d.xs[:, self.feat_bool], feat_lab=feat_lab, default=d)

log = logging.getLogger('@@DUMMY')
