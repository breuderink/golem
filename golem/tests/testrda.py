import unittest
import pylab
from ..data import gaussian_dataset
from ..nodes import RDA
from .. import plots, cv, loss

class TestRDA(unittest.TestCase):
  def setUp(self):
    self.d = gaussian_dataset([100, 100, 100])
    #self.d = gaussian_dataset([100, 20, 10])

  def test_qdc(self):
    c = RDA(alpha=0, beta=0)
    self.assert_(loss.mean_std(loss.accuracy, cv.rep_cv(self.d, c))[0] > .90)

  def test_visual_rda(self):
    c = RDA(alpha=0, beta=.5)
    c.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(c)
    pylab.savefig('rda.eps')


  def test_visual_qda(self):
    c = RDA(alpha=0, beta=0)
    c.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(c)
    pylab.savefig('qda.eps')

  def test_visual_lda(self):
    c = RDA(alpha=0, beta=1)
    c.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(c)
    pylab.savefig('lda.eps')

  def test_visual_nm(self):
    c = RDA(alpha=1, beta=0)
    c.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    pylab.axis('equal')
    plots.plot_classifier_hyperplane(c)
    pylab.savefig('nm.eps')
