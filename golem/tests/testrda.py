import unittest, os.path
import numpy as np
import pylab
from ..data import gaussian_dataset
from ..nodes import RDA, NMC, LDA, QDA
from .. import plots, cv, loss

class TestRDA(unittest.TestCase):
  def setUp(self):
    self.d = gaussian_dataset([100, 50, 100])

  def test_qdc(self):
    cl = QDA()
    self.assert_(loss.mean_std(loss.accuracy, cv.rep_cv(self.d, cl))[0] > .90)

  def test_visual_rda(self):
    cl = RDA(alpha=0, beta=.5)
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'rda.eps'))

  def test_visual_qda(self):
    cl = QDA()
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'qda.eps'))

  def test_visual_lda(self):
    cl = LDA()
    cl.train(self.d)

    pylab.clf()
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'lda.eps'))

  def test_visual_nm(self):
    cl = NMC()
    cl.train(self.d)

    pylab.clf()
    pylab.axis('equal')
    plots.plot_classifier(cl, self.d, densities=True, log_p=True)
    pylab.savefig(os.path.join('out', 'nm.eps'))
