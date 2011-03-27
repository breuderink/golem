import unittest, os.path
import numpy as np
import pylab
from ..data import gaussian_dataset
from ..nodes import RDA, NMC, LDA, QDA
from .. import plots, cv, perf

class TestRDA(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)
    self.d = gaussian_dataset([100, 50, 100])

  def test_qdc(self):
    cl = QDA(cov_f=np.cov)
    # test performance
    self.assert_(perf.mean_std(perf.accuracy, cv.rep_cv(self.d, cl))[0] > .95)

    # test probabilities
    td = cl.train_apply(self.d, self.d)
    np.testing.assert_array_less(td.X, 0) # p(x | y) <= 1

    # test quadratic boundary
    Si0 = cl.S_is[0]
    for i, Si in enumerate(cl.S_is):
      if i > 0:
        # inverse covariances are different
        self.assert_(np.linalg.norm(Si - Si0) > 0)
      # inverse covariance is not axis-aligned
      self.assert_(np.linalg.norm(Si - np.diag(np.diag(Si))) > 0)

  def test_lda(self):
    # Similar to QDA, but less complex. Only test linear boundaries.
    cl = LDA() 
    cl.train(self.d)
    Si0 = cl.S_is[0]
    for Si in cl.S_is:
      # same inverse covariance
      np.testing.assert_equal(Si, Si0) 
      # inverse covariance is not axis-aligned
      self.assert_(np.linalg.norm(Si - np.diag(np.diag(Si))) > 0)

  def test_nmc(self):
    # Similar to QDA, but less complex. Test for diagonal covariance.
    cl = NMC() 
    cl.train(self.d)
    Si0 = cl.S_is[0]
    for Si in cl.S_is:
      # same inverse covariance
      np.testing.assert_equal(Si, Si0) 
      # inverse covariance is axis-aligned
      self.assert_(np.linalg.norm(Si - np.diag(np.diag(Si))) < 1e-10)

  def test_visual_rda(self):
    cl = RDA(alpha=0, beta=.5)
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'rda.png'))

  def test_visual_qda(self):
    cl = QDA()
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'qda.png'))

  def test_visual_lda(self):
    cl = LDA()
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'lda.png'))

  def test_visual_nm(self):
    cl = NMC()
    cl.train(self.d)

    pylab.clf()
    pylab.axis('equal')
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'nm.png'))
