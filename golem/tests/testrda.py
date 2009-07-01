import unittest
import pylab
from ..data import gaussian_dataset
from ..nodes import RDA
from .. import plots, cv, loss

class TestRDA(unittest.TestCase):
  def setUp(self):
    self.d = gaussian_dataset([100, 5, 30])

  def test_qdc(self):
    c = RDA()
    self.assert_(loss.mean_std(loss.accuracy, cv.rep_cv(self.d, c))[0] > .95)

  def test_visual(self):
    c = RDA()
    c.train(self.d)

    pylab.clf()
    plots.scatter_plot(self.d)
    plots.plot_classifier_hyperplane(c)
    pylab.savefig('rda.eps')
