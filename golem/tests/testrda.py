import unittest
import numpy as np
import pylab
from ..data import gaussian_dataset
from ..nodes import RDA
from .. import plots, cv, loss

class TestRDA(unittest.TestCase):
  def setUp(self):
    self.d = gaussian_dataset([100, 50, 100])

  def test_qdc(self):
    cl = RDA(alpha=0, beta=0)
    self.assert_(loss.mean_std(loss.accuracy, cv.rep_cv(self.d, cl))[0] > .90)

  def test_visual_rda(self):
    cl = RDA(alpha=0, beta=.5)
    cl.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(cl)
    pylab.savefig('rda.eps')

  def test_visual_qda(self):
    cl = RDA(alpha=0, beta=0)
    cl.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(cl)
    pylab.savefig('qda.eps')

  def test_visual_lda(self):
    cl = RDA(alpha=0, beta=1)
    cl.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(cl)
    pylab.savefig('lda.eps')

  def test_visual_nm(self):
    cl = RDA(alpha=1, beta=0)
    cl.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    pylab.axis('equal')
    plots.plot_classifier_hyperplane(cl)
    pylab.savefig('nm.eps')
