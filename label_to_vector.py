from data import Data
from block import Block

class LabelToVector(Block):
  def __init__(self):
    Block.__init__(self)
    self.model.dictionary = {}
    self.name = 'Label converter'
      
  def train_method(self, data):
    assert(len(data.ys[0]) == 1) # lists cannot be hashed, we used one class-label
    labels = set([y[0] for y in data.ys])
    self.model.dictionary = dict(zip(labels, range(len(labels))))
    return self.test_method(data)
    
  def test_method(self, data):
    return Data(data.xs, [[self.model.dictionary[y[0]]] for y in data.ys], 
      data.signature + [self.name])

