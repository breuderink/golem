import numpy as np
from ..dataset import DataSet

class ZScore:
  def __init__(self):
    self.mean = self.std = None
  def train(self, d):
    self.mean = np.mean(d.xs, axis=0)
    self.std = np.std(d.xs, axis=0)

  def test(self, d):
    zxs = (d.xs - self.mean) / self.std # Careful, uses broadcasting!
    return DataSet(xs=zxs, default=d)

  def __str__(self):
    if self.mean <> None:
      return 'ZScore (mean=%s, std=%s)' % (self.mean, self.std)
    else:
      return 'ZScore (untrained)'

