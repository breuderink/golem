import logging
import numpy as np
from golem import DataSet, helpers

auc_log = logging.getLogger('AUCFilter')
class AUCFilter:
  def __init__(self, min_auc = .6, nfeatures=None):
    self.min_auc = min_auc
    self.nfeatures = nfeatures

  def train(self, d):
    assert(d.nclasses == 2)
    labels = d.ys[:, 0]
    min_auc = self.min_auc
    aucs = np.zeros(d.nfeatures)
    for fi in xrange(d.nfeatures):
      scores = d.xs[:, fi]
      aucs[fi] = helpers.auc(scores, labels)
    auc_log.debug('Found AUCs for features: %s' % aucs)

    if self.nfeatures == None:
      # threshold filter
      self.feat_bool = np.logical_or(aucs > min_auc, aucs < 1 - min_auc)
      if (self.feat_bool == False).all():
        auc_log.warning('No features found above threshold.')
    else:
      # pick best nfeatures
      self.feat_bool = np.zeros(aucs.size, bool)
      self.feat_bool[aucs.argsort()[:self.nfeatures]] = True

    auc_log.info('Keeping %d%% of the features' % (100 * self.feat_bool.mean()))
    self.cl_lab = d.cl_lab
    self.aucs = aucs

  def test(self, d):
    feat_lab = [d.feat_lab[i] for i in range(len(d.feat_lab)) \
      if self.feat_bool[i]]
    return DataSet(xs=d.xs[:, self.feat_bool], feat_lab=feat_lab, default=d)
