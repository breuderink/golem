import unittest, os.path
import numpy as np
from .. import plots, stat
import matplotlib.pyplot as plt

class TestPlots(unittest.TestCase):
  def setUp(self):
    np.random.seed(0)

  def tearDown(self):
    plt.close('all')

  def test_perf_scatter(self):
    plots.perf_scatter(np.linspace(0, .5, 20), np.linspace(0, .7, 20))
    plt.xlabel('low'); plt.ylabel('high')
    plt.savefig(os.path.join('out', 'perf_scatter.png'))

  def test_roc(self):
    for noise in [.3, .1]:
      tps, fps = stat.roc(
        (np.linspace(0, 1, 100) + noise * np.random.randn(100)),
        np.linspace(0, 1, 100) > .75)
      plots.plot_roc(tps, fps)

    plt.xlabel('low'); plt.ylabel('high')
    plt.legend(['bad', 'good'])
    plt.savefig(os.path.join('out', 'roc.png'))
    plt.close()
