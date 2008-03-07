from data import Data
from block import Block

class Chain(Block):
  def __init__(self, block_list):
    Block.__init__(self)
    self.name = 'Chain'
    self.model.block_list = block_list
      
  def train(self, data, trace):
    for block in self.model.block_list:
      (data, trace) = block.train(data, trace)
    return (data, trace)
    
  def test(self, data, trace):
    for block in self.model.block_list:
      (data, trace) = block.test(data, trace)
    return (data, trace)
    
  def __str__(self):
    return 'Chain'
