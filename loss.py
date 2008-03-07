import itertools
from data import Data
from block import Block

class Loss(Block):
  def __init__(self):
    Block.__init__(self)
    self.name = 'Loss'
    self.model.class_loss = None
      
  def train_method(self, data):
    return data
    
  def test_method(self, data):
    errors = filter(lambda (x, y): x != y, itertools.izip(data.xs, data.ys))
    self.model.class_loss = len(errors)/float(len(data.xs))
    return data    
    
